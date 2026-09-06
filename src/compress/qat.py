"""Quantization-aware fine-tuning and prune-and-retrain.

Post-training compression alone loses accuracy that fine-tuning recovers, and
the recovery grows as bit width falls: at 8 bits it is a fraction of a point, at
4 bits it is the difference between a usable and an unusable model. Deep
Compression makes the same point - its pruning and its trained quantization are
both *retraining* stages, not one-shot transforms.

Two mechanisms, both hand-written:

1. Prune-and-retrain. The pruning mask is fixed, and the surviving weights are
   retrained to compensate for the removed ones. The mask is re-applied after
   every optimiser step, because momentum and weight decay both push pruned
   weights away from zero and a gradient-only mask would not hold them there.

2. Quantization-aware training via the straight-through estimator. The classic
   master-weight formulation:

       save   w_master
       w     <- fake_quantize(w_master)      quantize in place
       forward / backward                    gradients at the quantized point
       w     <- w_master                     restore full precision
       optimiser.step()                      update the master copy

   The gradient is evaluated where the network actually operates (on the
   quantized grid) but accumulated into a full-precision master, so updates
   smaller than one quantization step are not lost. Without the master copy,
   every update below half a step would round straight back and training would
   stall immediately.

   This is exactly the STE, applied at tensor granularity, and it needs no
   module surgery - which matters because a Parameter cannot be assigned a
   non-leaf tensor.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..evaluate import evaluate
from ..utils import AverageMeter
from .pipeline import (CompressionConfig, attach_activation_quantizers,
                       calibrate_activations)
from .prune import PruneConfig, Pruner
from .quantize import fake_quantize
from .weight_share import share_weights


@dataclass
class QATConfig:
    """Fine-tuning schedule for the compressed model."""

    epochs: int = 15
    lr: float = 0.01                 # ~1/10 of the baseline LR: we are recovering,
                                     # not re-training, so large steps destroy the
                                     # solution the pruning mask was chosen from.
    momentum: float = 0.9
    weight_decay: float = 0.0        # off: decay fights the fixed quantization grid
                                     # and pulls surviving weights toward zero, which
                                     # the mask has already accounted for.
    label_smoothing: float = 0.1
    min_lr: float = 0.0
    # Global gradient-norm clip. This is not optional at low bit width.
    # The straight-through estimator evaluates the gradient at the quantized
    # point but applies it to a full-precision master, and at 3 bits the
    # mismatch produces very large gradients: an instrumented run reached
    # |grad| = 8.3e5 by iteration 25, drove one layer's weights to 6.7e14, and
    # the loss pinned at ln(10) + smoothing (uniform output) for the rest of
    # training. The failure is absorbing - a dead layer has no gradient, so no
    # later step can revive it - and whether it triggers depends on batch order,
    # so identical configurations succeed or fail run to run. Clipping removes
    # the run-to-run lottery. See scripts/diagnose_qat_collapse.py.
    grad_clip: float = 5.0
    # Re-run k-means every `recluster_every` epochs so the codebook tracks the
    # weights as they move. 0 disables re-clustering (codebook fixed after the
    # first fit).
    recluster_every: int = 1
    quantize_weights: bool = True
    quantize_activations: bool = True


def _layer_policy(model: nn.Module, cfg: CompressionConfig) -> Dict[str, Dict]:
    """Per-layer bit width and prunability, matching pipeline.compress_weights.

    Kept consistent with the pipeline's policy so that the model fine-tuned here
    is the same model the pipeline later measures. A divergence between the two
    would produce an accuracy number for one configuration and a size number for
    another.
    """
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    first, last = convs[0][0], convs[-1][0]
    policy = {}
    for name, m in convs:
        if name in (first, last):
            bits, prunable = cfg.edge_bits, False
        elif isinstance(m, nn.Conv2d) and m.groups > 1:
            bits = cfg.depthwise_bits if cfg.depthwise_bits is not None else cfg.weight_bits
            prunable = False
        else:
            bits, prunable = cfg.weight_bits, True
        policy[name] = {"module": m, "bits": bits, "prunable": prunable}
    return policy


@torch.no_grad()
def _quantize_in_place(policy: Dict[str, Dict], masks: Dict[str, torch.Tensor],
                       cfg: CompressionConfig, codebooks: Optional[Dict] = None
                       ) -> Dict[str, torch.Tensor]:
    """Replace every weight with its quantized value, returning the fp32 masters."""
    masters = {}
    for name, p in policy.items():
        m, bits = p["module"], p["bits"]
        masters[name] = m.weight.data.clone()
        mask = masks.get(name)

        if codebooks is not None and name in codebooks:
            # k-means path: assignments and centroids are fixed for this step,
            # so the forward pass sees exactly the deployed codebook.
            centroids, indices = codebooks[name]
            q = centroids[indices].reshape(m.weight.shape).to(m.weight.dtype)
            if mask is not None:
                q = q * mask
            m.weight.data = q
        else:
            # Granularity must match what the pipeline will actually deploy.
            # Hard-coding per-channel here would train the network against one
            # quantizer and then measure it under another - a train/deploy
            # mismatch that costs accuracy silently, with no error anywhere.
            gran = "per_tensor" if cfg.weight_method == "linear_per_tensor" else "per_channel"
            q = fake_quantize(m.weight.data, bits, "symmetric", gran, channel_dim=0)
            if mask is not None:
                q = q * mask
            m.weight.data = q
    return masters


@torch.no_grad()
def _restore(policy: Dict[str, Dict], masters: Dict[str, torch.Tensor]) -> None:
    """Put the full-precision master weights back before the optimiser step."""
    for name, p in policy.items():
        p["module"].weight.data = masters[name]


@torch.no_grad()
def _refit_codebooks(policy: Dict[str, Dict], masks: Dict[str, torch.Tensor],
                     cfg: CompressionConfig) -> Dict[str, tuple]:
    """Re-run k-means on the current master weights.

    Refitting periodically lets the codebook follow the weights as fine-tuning
    moves them. Holding the codebook fixed from the first epoch would make the
    optimiser chase a grid that no longer matches the distribution it is
    shaping.
    """
    books = {}
    for name, p in policy.items():
        m, bits = p["module"], p["bits"]
        mask = masks.get(name, torch.ones_like(m.weight, dtype=torch.bool))
        res = share_weights(m.weight.data, bits, mask=mask, init=cfg.kmeans_init)
        books[name] = (res.centroids, res.indices)
    return books


def finetune(model: nn.Module, cfg: CompressionConfig, qcfg: QATConfig,
             train_loader, test_loader, calib_loader, device: torch.device,
             log_fn=None) -> tuple:
    """Prune, then fine-tune the model with quantization in the loop.

    Returns (fine-tuned model, history). The input model is not modified.

    Sequence:
      1. compute pruning masks from the trained weights and apply them,
      2. attach and calibrate the activation quantizers, so fine-tuning adapts
         to activation quantization as well as weight quantization,
      3. fine-tune with the master-weight STE described in the module docstring,
         re-applying the mask after every optimiser step.
    """
    work = copy.deepcopy(model).to(device)
    policy = _layer_policy(work, cfg)

    # ---- stage 1: pruning
    masks: Dict[str, torch.Tensor] = {}
    pruner = None
    if cfg.sparsity > 0:
        pruner = Pruner(work, PruneConfig(sparsity=cfg.sparsity, scope=cfg.prune_scope,
                                          max_layer_sparsity=cfg.max_layer_sparsity))
        pruner.compute_masks()
        pruner.apply()
        masks = pruner.masks

    # ---- activation quantizers, calibrated on held-out training data
    quantizers = {}
    if qcfg.quantize_activations and cfg.quantize_activations:
        quantizers, _ = attach_activation_quantizers(work, cfg)
        calibrate_activations(work, quantizers, calib_loader, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=qcfg.label_smoothing)
    optimizer = torch.optim.SGD(work.parameters(), lr=qcfg.lr,
                                momentum=qcfg.momentum, weight_decay=qcfg.weight_decay)

    # Only the codebook methods need a fitted codebook; the linear methods derive
    # their grid from the weights on every step.
    use_kmeans = cfg.weight_method in ("kmeans", "ecsq", "auto")
    codebooks = _refit_codebooks(policy, masks, cfg) if (use_kmeans and qcfg.quantize_weights) else None

    history: List[Dict] = []
    iters = len(train_loader)

    for epoch in range(qcfg.epochs):
        if codebooks is not None and qcfg.recluster_every and epoch > 0 \
                and epoch % qcfg.recluster_every == 0:
            codebooks = _refit_codebooks(policy, masks, cfg)

        work.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        for it, (images, targets) in enumerate(train_loader):
            # cosine schedule over the fine-tuning horizon
            prog = (epoch + it / iters) / max(qcfg.epochs, 1e-8)
            lr = qcfg.min_lr + 0.5 * (qcfg.lr - qcfg.min_lr) * (1 + math.cos(math.pi * prog))
            for g in optimizer.param_groups:
                g["lr"] = lr

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # --- STE: quantize -> forward/backward -> restore master -> step
            masters = _quantize_in_place(policy, masks, cfg, codebooks) \
                if qcfg.quantize_weights else None

            optimizer.zero_grad(set_to_none=True)
            logits = work(images)
            loss = criterion(logits, targets)
            loss.backward()

            if qcfg.grad_clip and qcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(work.parameters(), qcfg.grad_clip)

            if masters is not None:
                _restore(policy, masters)
            optimizer.step()

            # Re-apply the mask: momentum and weight decay both move pruned
            # weights away from zero, so masking gradients alone is not enough.
            if pruner is not None:
                pruner.apply()

            n = targets.size(0)
            loss_meter.update(loss.item(), n)
            acc_meter.update((logits.argmax(1) == targets).float().mean().item() * 100, n)

        # Evaluate in the deployed configuration: weights quantized, mask applied.
        masters = _quantize_in_place(policy, masks, cfg, codebooks) \
            if qcfg.quantize_weights else None
        te = evaluate(work, test_loader, device)
        if masters is not None:
            _restore(policy, masters)

        rec = {"epoch": epoch + 1, "lr": lr, "train_loss": loss_meter.avg,
               "train_top1": acc_meter.avg, "test_top1": te["top1"], "test_loss": te["loss"]}
        history.append(rec)
        if log_fn:
            log_fn(rec)

    # Leave the model in its deployed state: quantized weights, mask applied.
    if qcfg.quantize_weights:
        _quantize_in_place(policy, masks, cfg, codebooks)
    return work, history

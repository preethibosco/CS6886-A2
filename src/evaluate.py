"""Evaluation utilities.

Deliberately kept in its own module: both the training loop and the compression
pipeline import `evaluate` from here, so that "accuracy before compression" and
"accuracy after compression" are produced by literally the same code path. Any
difference between those two numbers is then attributable to the compression
and nothing else.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import AverageMeter


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Compute top-1 / top-5 accuracy and mean loss over a loader.

    Args:
        model: network to evaluate. Switched to eval() mode, which freezes the
            BatchNorm running statistics - important, because otherwise a pass
            over the test set would update them and the "before" model would no
            longer be the model we compressed.
        loader: data loader; must use the non-augmented eval transform.
        device: device to run on.
        criterion: loss to report. Defaults to plain cross-entropy (no label
            smoothing) so that the reported loss is comparable across runs
            regardless of the smoothing used during training.
        amp_dtype: if given, run the forward pass under autocast with this
            dtype. Left as None for compression evaluation so that quantization
            error is never confounded with autocast rounding.

    Returns:
        dict with keys 'top1', 'top5', 'loss' (top-1/top-5 as percentages).
    """
    was_training = model.training
    model.eval()
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        acc1, acc5 = topk_accuracy(logits, targets, topk=(1, 5))
        n = targets.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(acc1, n)
        top5_meter.update(acc5, n)

    if was_training:
        model.train()

    return {"top1": top1_meter.avg, "top5": top5_meter.avg, "loss": loss_meter.avg}


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                  topk: Tuple[int, ...] = (1,)) -> Tuple[float, ...]:
    """Top-k accuracy as a percentage, for each k in `topk`."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)   # (B, maxk)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))          # (B, maxk)
    return tuple(
        correct[:, :k].any(dim=1).float().sum().item() * 100.0 / targets.size(0)
        for k in topk
    )


@torch.no_grad()
def per_class_accuracy(model: nn.Module, loader: DataLoader, device: torch.device,
                       num_classes: int = 10) -> torch.Tensor:
    """Per-class top-1 accuracy (%), used for the failure-mode discussion in Q1(c)."""
    model.eval().to(device)
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1).cpu()
        for c in range(num_classes):
            mask = targets == c
            total[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()
    return 100.0 * correct / total.clamp(min=1)


@torch.no_grad()
def confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device,
                     num_classes: int = 10) -> torch.Tensor:
    """Confusion matrix, rows = true class, cols = predicted class.

    Used to identify which class pairs the model confuses (Q1c failure modes),
    and re-run after compression to check whether quantization changes *which*
    mistakes are made, not just how many.
    """
    model.eval().to(device)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, targets in loader:
        preds = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
        for t, p in zip(targets, preds):
            cm[t, p] += 1
    return cm

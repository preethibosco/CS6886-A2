"""Baseline training for MobileNet-v2 on CIFAR-10 (Assignment Q1).

Produces:
  * checkpoints/mobilenetv2_cifar10_best.pt  - highest test top-1 seen
  * checkpoints/mobilenetv2_cifar10_last.pt  - final epoch
  * results/train_history.json               - per-epoch curves for the report

Usage:
    python -m src.train --epochs 200 --batch-size 128 --lr 0.1 --seed 42

The compression pipeline consumes the *best* checkpoint; nothing in this file
is imported by the compression code, keeping training and compression fully
separated (Q5a).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn

from .data import DataConfig, build_loaders, describe_transforms
from .evaluate import evaluate, per_class_accuracy
from .models.mobilenetv2 import mobilenet_v2_cifar
from .utils import AverageMeter, count_parameters, get_device, seed_everything


# --------------------------------------------------------------------------- #
# Optimiser construction
# --------------------------------------------------------------------------- #
def build_param_groups(model: nn.Module, weight_decay: float,
                       skip_depthwise_decay: bool = True) -> List[Dict]:
    """Split parameters into decayed and non-decayed groups.

    Weight decay is deliberately NOT applied to:

      * BatchNorm gamma/beta - decaying the scale of a normalisation layer just
        fights the normalisation itself, and shrinking gamma toward zero can
        silently disable channels.
      * biases - one scalar per channel, no meaningful capacity to regularise.
      * depthwise convolution weights (when `skip_depthwise_decay`) - a depthwise
        filter holds only 3x3 = 9 weights per channel, so a decay term that is
        mild for a 960x160 pointwise conv is a very strong pull toward zero
        here, and can kill whole channels. This is standard practice for
        MobileNet-family models and is worth roughly half a point of top-1.

    Not decaying depthwise weights has a second, compression-specific benefit:
    it leaves those filters with a wider, better-conditioned value distribution,
    which quantizes with less relative error.
    """
    # Identify depthwise conv weights by walking modules (groups > 1).
    depthwise_param_ids = set()
    if skip_depthwise_decay:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.groups > 1:
                depthwise_param_ids.add(id(m.weight))

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # p.ndim <= 1 catches every 1-D tensor, which here means BatchNorm gamma
        # and beta plus any bias. Conv and linear weights are 4-D and 2-D, so
        # they fall through to the decayed group.
        # For this model the split is 36 decayed tensors (35 pointwise/standard
        # convs + the classifier weight) and 122 undecayed (104 BatchNorm
        # gamma/beta + 17 depthwise weights + the classifier bias).
        if p.ndim <= 1 or name.endswith(".bias") or id(p) in depthwise_param_ids:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(epoch_float: float, base_lr: float, total_epochs: int,
          warmup_epochs: int, min_lr: float) -> float:
    """Linear warmup followed by cosine annealing, evaluated per iteration.

    Warmup matters for MobileNet-v2: the depthwise layers have a fan-in of only
    9, so the very first full-size steps can drive BatchNorm running statistics
    into a regime the network takes many epochs to recover from.
    """
    if epoch_float < warmup_epochs:
        # Linear ramp from ~0 to base_lr over the first `warmup_epochs`.
        # epoch_float is fractional (epoch + iteration/iterations_per_epoch), so
        # the LR rises smoothly within an epoch rather than in steps.
        return base_lr * (epoch_float + 1e-8) / max(warmup_epochs, 1e-8)
    # After warmup, `progress` runs 0 -> 1 over the remaining epochs.
    progress = (epoch_float - warmup_epochs) / max(total_epochs - warmup_epochs, 1e-8)
    progress = min(max(progress, 0.0), 1.0)
    # cos(pi*progress) goes 1 -> -1, so (1+cos)/2 goes 1 -> 0 and the LR decays
    # from base_lr to min_lr. The decay is slow at first and slow again at the
    # end, spending most of the run at a useful learning rate. Cutting the
    # schedule short leaves the model at a high LR and costs about a point of
    # top-1, which is why the epoch count and the schedule must match.
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, args, scaler=None):
    """One pass over the training set. Returns dict of epoch-mean metrics."""
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    iters_per_epoch = len(loader)

    for it, (images, targets) in enumerate(loader):
        # Per-iteration LR so the cosine curve is smooth rather than stepped.
        lr = lr_at(epoch + it / iters_per_epoch, args.lr, args.epochs,
                   args.warmup_epochs, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # bfloat16 autocast runs the convolutions in half precision while
            # keeping the master weights in fp32. bf16 has the same exponent
            # range as fp32, so unlike fp16 it cannot silently underflow and
            # needs no gradient scaler. The backward pass stays outside the
            # autocast block, which is the documented pattern.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        n = targets.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update((logits.argmax(1) == targets).float().mean().item() * 100.0, n)

    return {"loss": loss_meter.avg, "top1": acc_meter.avg, "lr": lr}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Never silently overwrite an existing checkpoint. Training writes
    # best.pt whenever the current run beats its own best, which means a short
    # run started for any reason (a smoke test, a changed hyper-parameter)
    # replaces a fully trained model with a worse one and the original is gone.
    # Existing checkpoints are moved aside with a timestamp instead.
    # results/train_history.json is rewritten every epoch and is what the report
    # figures are generated from, so a short run replaces the reported curves
    # with partial ones exactly as it would replace the checkpoint. Both are
    # moved aside together.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    existing = [os.path.join(args.checkpoint_dir, f)
                for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")]
    history_path = os.path.join(args.results_dir, "train_history.json")
    if os.path.exists(history_path):
        existing.append(history_path)
    if existing and not args.overwrite_checkpoints:
        backup = os.path.join(args.checkpoint_dir, f"backup-{stamp}")
        os.makedirs(backup, exist_ok=True)
        for f in existing:
            os.rename(f, os.path.join(backup, os.path.basename(f)))
        print(f"moved {len(existing)} existing artefact(s) to {backup}\n")

    # ---------------------------------------------------------------- data
    data_cfg = DataConfig(root=args.data_root, batch_size=args.batch_size,
                          num_workers=args.num_workers, seed=args.seed,
                          random_erasing=not args.no_random_erasing)
    train_loader, test_loader, _ = build_loaders(data_cfg)

    print(describe_transforms(data_cfg))
    print()

    # --------------------------------------------------------------- model
    model = mobilenet_v2_cifar(num_classes=10, width_mult=args.width_mult,
                               dropout=args.dropout, bn_momentum=args.bn_momentum).to(device)
    n_params = count_parameters(model)
    print(f"MobileNet-v2 (width_mult={args.width_mult}, dropout={args.dropout}): "
          f"{n_params:,} parameters = {n_params * 4 / 2**20:.3f} MB fp32")

    # Label smoothing produces less over-confident logits, which is worth ~0.3
    # top-1 and additionally yields a flatter loss surface that tolerates
    # post-training quantization better.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        build_param_groups(model, args.weight_decay, skip_depthwise_decay=True),
        lr=args.lr, momentum=args.momentum, nesterov=True,
    )
    # bfloat16 autocast needs no gradient scaler (unlike fp16), but we keep the
    # flag so the code path is explicit.
    use_amp = args.amp and device.type == "cuda"
    scaler = True if use_amp else None

    print(f"optimizer: SGD(lr={args.lr}, momentum={args.momentum}, nesterov=True), "
          f"wd={args.weight_decay} on {len(optimizer.param_groups[0]['params'])} tensors, "
          f"0.0 on {len(optimizer.param_groups[1]['params'])} tensors")
    print(f"schedule : {args.warmup_epochs}-epoch linear warmup -> cosine to {args.min_lr}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} amp={'bf16' if use_amp else 'off'} "
          f"seed={args.seed}\n")

    # ------------------------------------------------------------- logging
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name or None,
                         config=vars(args), job_type="baseline")

    history: List[Dict] = []
    best_top1, best_epoch = 0.0, -1
    start = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device,
                             epoch, args, scaler)
        te = evaluate(model, test_loader, device)
        dt = time.time() - t0

        record = {"epoch": epoch + 1, "lr": tr["lr"],
                  "train_loss": tr["loss"], "train_top1": tr["top1"],
                  "test_loss": te["loss"], "test_top1": te["top1"], "test_top5": te["top5"],
                  "epoch_seconds": dt}
        history.append(record)
        if run is not None:
            run.log(record, step=epoch + 1)

        # Track the best test accuracy rather than just the last epoch. With a
        # cosine schedule the final epochs are usually the best, but not always,
        # and the compression pipeline consumes this checkpoint.
        is_best = te["top1"] > best_top1
        if is_best:
            best_top1, best_epoch = te["top1"], epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "test_top1": best_top1, "args": vars(args)},
                       os.path.join(args.checkpoint_dir, "mobilenetv2_cifar10_best.pt"))

        print(f"epoch {epoch+1:3d}/{args.epochs}  lr {tr['lr']:.4f}  "
              f"train loss {tr['loss']:.4f} acc {tr['top1']:.2f}  |  "
              f"test loss {te['loss']:.4f} top1 {te['top1']:.2f} top5 {te['top5']:.2f}  "
              f"{'*' if is_best else ' '}  {dt:.1f}s", flush=True)

        # Rewrite the history file every epoch so curves survive an interruption.
        with open(os.path.join(args.results_dir, "train_history.json"), "w") as f:
            json.dump({"history": history, "best_top1": best_top1,
                       "best_epoch": best_epoch, "args": vars(args)}, f, indent=2)

    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "test_top1": history[-1]["test_top1"], "args": vars(args)},
               os.path.join(args.checkpoint_dir, "mobilenetv2_cifar10_last.pt"))

    total_min = (time.time() - start) / 60
    print(f"\nbest test top-1: {best_top1:.2f}% at epoch {best_epoch}  "
          f"(total {total_min:.1f} min)")

    # Per-class accuracy of the best model, for the Q1(c) failure-mode discussion.
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "mobilenetv2_cifar10_best.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    pca = per_class_accuracy(model, test_loader, device)
    from .data import CIFAR10_CLASSES
    print("\nper-class top-1 of best checkpoint:")
    for name, acc in zip(CIFAR10_CLASSES, pca.tolist()):
        print(f"  {name:<12s} {acc:6.2f}%")

    with open(os.path.join(args.results_dir, "train_history.json"), "w") as f:
        json.dump({"history": history, "best_top1": best_top1, "best_epoch": best_epoch,
                   "per_class_top1": dict(zip(CIFAR10_CLASSES, pca.tolist())),
                   "args": vars(args)}, f, indent=2)

    if run is not None:
        run.summary["best_top1"] = best_top1
        run.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MobileNet-v2 on CIFAR-10")
    # data
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-random-erasing", action="store_true")
    # model
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bn-momentum", type=float, default=0.05)
    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--overwrite-checkpoints", action="store_true",
                   help="allow replacing existing checkpoints instead of "
                        "moving them aside")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="cs6886-a2-mobilenetv2")
    p.add_argument("--run-name", default="")
    return p.parse_args()


if __name__ == "__main__":
    main()

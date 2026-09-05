"""Training curves and failure-mode figures for Q1(c).

Reads results/train_history.json (written every epoch by src/train.py) and
produces:
  results/figures/fig_training_curves.png  - loss and top-1 vs epoch, plus LR
  results/figures/fig_per_class.png        - per-class accuracy and confusion

Run:  python scripts/plot_curves.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import CIFAR10_CLASSES, DataConfig, build_loaders
from src.evaluate import confusion_matrix, per_class_accuracy
from src.models.mobilenetv2 import mobilenet_v2_cifar
from src.utils import get_device

ap = argparse.ArgumentParser()
ap.add_argument("--history", default="results/train_history.json")
ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
ap.add_argument("--outdir", default="results/figures")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

hist = json.load(open(args.history))
H = hist["history"]
ep = [r["epoch"] for r in H]

# ---------------------------------------------------------------- curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

axes[0].plot(ep, [r["train_loss"] for r in H], label="train", lw=1.4)
axes[0].plot(ep, [r["test_loss"] for r in H], label="test", lw=1.4)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, [r["train_top1"] for r in H], label="train", lw=1.4)
axes[1].plot(ep, [r["test_top1"] for r in H], label="test", lw=1.4)
best = hist.get("best_top1", max(r["test_top1"] for r in H))
best_ep = hist.get("best_epoch", ep[max(range(len(H)), key=lambda i: H[i]["test_top1"])])
axes[1].axhline(best, ls="--", c="gray", lw=1)
axes[1].annotate(f"best {best:.2f}% @ epoch {best_ep}", (best_ep, best),
                 textcoords="offset points", xytext=(-10, -18), fontsize=9)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("top-1 accuracy (%)")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(ep, [r["lr"] for r in H], lw=1.4, c="C2")
axes[2].set_xlabel("epoch"); axes[2].set_ylabel("learning rate")
axes[2].set_title("LR schedule (warmup + cosine)"); axes[2].grid(alpha=0.3)

# The train/test gap is the overfitting signal discussed in the report.
gap = H[-1]["train_top1"] - H[-1]["test_top1"]
fig.suptitle(f"MobileNet-v2 / CIFAR-10 baseline  -  final train-test gap {gap:.2f} pts", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(args.outdir, "fig_training_curves.png"), dpi=150, bbox_inches="tight")
print(f"wrote {args.outdir}/fig_training_curves.png")

# ---------------------------------------------------- per-class + confusion
if os.path.exists(args.checkpoint):
    device = get_device()
    model = mobilenet_v2_cifar().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)["model"])
    _, test_loader, _ = build_loaders(DataConfig(), download=False)
    pca = per_class_accuracy(model, test_loader, device).tolist()
    cm = confusion_matrix(model, test_loader, device).float()
    cmn = (cm / cm.sum(1, keepdim=True)).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = sorted(range(10), key=lambda i: pca[i])
    axes[0].barh([CIFAR10_CLASSES[i] for i in order], [pca[i] for i in order], color="C0")
    axes[0].axvline(sum(pca) / 10, ls="--", c="gray", lw=1,
                    label=f"mean {sum(pca)/10:.2f}%")
    axes[0].set_xlabel("top-1 accuracy (%)"); axes[0].set_xlim(min(pca) - 5, 100)
    axes[0].set_title("Per-class accuracy"); axes[0].legend()

    im = axes[1].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(10)); axes[1].set_yticks(range(10))
    axes[1].set_xticklabels(CIFAR10_CLASSES, rotation=90, fontsize=8)
    axes[1].set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
    axes[1].set_title("Confusion matrix (row-normalised)")
    for i in range(10):
        for j in range(10):
            if i != j and cmn[i, j] > 0.04:
                axes[1].text(j, i, f"{100*cmn[i,j]:.0f}", ha="center", va="center",
                             fontsize=7, color="darkred")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_per_class.png"), dpi=150, bbox_inches="tight")
    print(f"wrote {args.outdir}/fig_per_class.png")

    worst = sorted(range(10), key=lambda i: pca[i])[:3]
    print("\nfailure modes (Q1c):")
    for i in worst:
        conf = sorted(((cmn[i, j], j) for j in range(10) if j != i), reverse=True)[0]
        print(f"  {CIFAR10_CLASSES[i]:<12} {pca[i]:.2f}%  most confused with "
              f"{CIFAR10_CLASSES[conf[1]]} ({100*conf[0]:.1f}%)")

"""CIFAR-10 data pipeline.

Exposes three loaders:
  * train       - augmented, shuffled, used for gradient updates
  * test        - deterministic, used for every reported top-1 number
  * calibration - a held-out slice of the *train* set, used only to observe
                  activation ranges when calibrating the activation quantizer

The calibration split exists so that activation clipping thresholds are never
fitted on the test set. Fitting them on test would leak the evaluation data into
the compression pipeline and inflate the reported post-compression accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Per-channel statistics of the CIFAR-10 *training* split (RGB order).
# These are dataset-specific; substituting ImageNet statistics leaves the input
# distribution off-centre and costs roughly half a point of top-1.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


@dataclass
class DataConfig:
    """All knobs that affect the data pipeline, so a run is fully described by it."""

    root: str = "./data"
    batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 8
    # Number of *training* images set aside for activation-range calibration.
    calib_size: int = 1024
    # Random erasing is an extra regulariser applied after normalisation. It is
    # cheap and buys ~0.3-0.5 top-1 on CIFAR-10 with MobileNet-v2.
    random_erasing: bool = True
    erasing_prob: float = 0.25
    seed: int = 42


def build_transforms(cfg: DataConfig):
    """Return (train_transform, eval_transform).

    Train-time augmentation, in order:
      1. RandomCrop(32, padding=4) - pads 4px on every side then crops back to
         32x32. Gives translation invariance. This is the single most important
         augmentation on CIFAR-10; without it MobileNet-v2 memorises the train
         set (~99% train / ~88% test).
      2. RandomHorizontalFlip() - mirror invariance. Valid for all ten CIFAR-10
         classes (none of them are chiral, unlike digits or text).
      3. ToTensor() - HWC uint8 [0,255] -> CHW float [0,1].
      4. Normalize(mean, std) - per-channel standardisation to ~zero mean/unit
         variance, which keeps the first conv's pre-activations in a well
         conditioned range.
      5. RandomErasing() - optional; blanks a random rectangle *after*
         normalisation, so the erased region is filled with the channel mean
         rather than with black.

    Eval-time: ToTensor + Normalize only. Applying augmentation at eval time
    would make the reported top-1 a noisy estimate of the wrong quantity.
    """
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if cfg.random_erasing:
        train_ops.append(
            transforms.RandomErasing(p=cfg.erasing_prob, scale=(0.02, 0.20), value="random")
        )

    eval_ops = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_loaders(cfg: DataConfig, download: bool = True):
    """Build the train / test / calibration loaders.

    The calibration subset is drawn from the training split with the *eval*
    transform applied: we want the activation ranges the quantizer sees at
    inference time, not the ranges induced by random crops and erasing.
    """
    train_tf, eval_tf = build_transforms(cfg)

    train_set = CIFAR10(root=cfg.root, train=True, download=download, transform=train_tf)
    test_set = CIFAR10(root=cfg.root, train=False, download=download, transform=eval_tf)
    # Second view of the training data with no augmentation, for calibration.
    calib_source = CIFAR10(root=cfg.root, train=True, download=False, transform=eval_tf)

    # A dedicated generator, seeded explicitly, so the calibration images are
    # the same 1024 on every run regardless of what else consumed randomness
    # first. Activation ranges depend on which images are seen, so an unseeded
    # draw here would make the compression numbers wobble between runs.
    generator = torch.Generator().manual_seed(cfg.seed)
    calib_indices = torch.randperm(len(calib_source), generator=generator)[: cfg.calib_size]
    # Subset wraps the dataset and remaps indices; it copies no image data.
    calib_set = Subset(calib_source, calib_indices.tolist())

    common = dict(num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.eval_batch_size, shuffle=False, **common
    )
    calib_loader = DataLoader(
        calib_set, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader, calib_loader


def describe_transforms(cfg: DataConfig) -> str:
    """Human-readable transform listing, for the report (Q1a asks us to specify these)."""
    train_tf, eval_tf = build_transforms(cfg)
    lines = ["Train transforms:"]
    lines += [f"  {i+1}. {t}" for i, t in enumerate(train_tf.transforms)]
    lines += ["Eval transforms:"]
    lines += [f"  {i+1}. {t}" for i, t in enumerate(eval_tf.transforms)]
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = DataConfig()
    print(describe_transforms(cfg))
    tr, te, ca = build_loaders(cfg)
    xb, yb = next(iter(tr))
    print(f"\ntrain batches={len(tr)} test batches={len(te)} calib images={len(ca.dataset)}")
    print(f"batch shape={tuple(xb.shape)} dtype={xb.dtype} label range=[{yb.min()},{yb.max()}]")
    print(f"normalised batch mean={xb.mean():.4f} std={xb.std():.4f}")

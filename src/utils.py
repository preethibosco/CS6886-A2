"""Shared utilities: seeding, device selection, metric bookkeeping.

Kept deliberately free of any model/compression imports so that both the
training and the compression entry points can depend on it without creating
a cycle.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every RNG that can affect a run.

    Args:
        seed: the seed applied to python, numpy and torch (CPU + all CUDA devices).
        deterministic: if True, force cuDNN into deterministic algorithm selection.
            This makes runs bit-reproducible at a ~10-20% throughput cost, so we
            leave it off for the long baseline training run and switch it on for
            the (short) compression evaluations where reproducibility is what we
            are actually reporting.

    Note on remaining nondeterminism: DataLoader worker processes are seeded from
    the parent, but atomicAdd-based CUDA kernels remain nondeterministic unless
    `deterministic=True`. Documented here because Q5(b) asks for seed configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some conv/pool backward kernels need this to pick deterministic paths.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # benchmark=True lets cuDNN autotune conv algorithms for our fixed 32x32
        # input shape, which is a meaningful speedup for depthwise convolutions.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the requested device, falling back to CPU if CUDA is unavailable."""
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


class AverageMeter:
    """Running mean of a scalar, weighted by batch size.

    Used for loss and accuracy so that the epoch-level number is the true mean
    over samples rather than the mean over batches (these differ when the last
    batch is smaller).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Total number of scalar parameters in a model."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)

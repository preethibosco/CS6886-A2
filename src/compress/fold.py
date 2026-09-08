"""BatchNorm folding: absorb BN into the preceding convolution.

At inference BatchNorm is an affine map with frozen statistics, so it can be
folded exactly into the convolution before it:

    y = gamma * (W*x - mu) / sqrt(var + eps) + beta
      = (gamma / sqrt(var + eps)) * W * x  +  (beta - gamma*mu / sqrt(var + eps))

so the folded convolution is

    W' = W * s,   s = gamma / sqrt(var + eps)     (one scale per output channel)
    b' = beta - s * mu

The transform is exact - the folded network computes the same function to
floating-point precision - and it is worth doing because BatchNorm is 18.2% of
the compressed model at w3/sparsity 0.8, and it does not compress: measured
end-to-end, 8 -> 6 bit BatchNorm costs about 2 points of top-1 and 8 -> 4 bit
destroys the network entirely (19.2%). Coarse quantization of `running_var`
breaks the normalisation itself.

Folding replaces 4 stored vectors per layer (gamma, beta, running_mean,
running_var) with 1 (the fused bias), a 4x reduction in BatchNorm storage.

The catch, and the reason this is measured rather than assumed: folding
multiplies each output channel of W by its own scale s, which widens the spread
of per-channel weight ranges. Per-tensor quantization derives its step from
max|W| over the whole tensor, so a wider spread means the small-scale channels
lose resolution. Folding therefore helps storage and may hurt accuracy, and
which effect wins is an empirical question.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a new Conv2d with `bn` folded into it. Does not modify the inputs."""
    # s is BatchNorm's per-channel multiplier once the statistics are frozen.
    # gamma is the learned scale, running_var the tracked variance, eps a small
    # constant that stops a near-zero-variance channel dividing by zero.
    # One value per output channel, so folding it in rescales each filter
    # independently. That is exactly why folding widens the spread of
    # per-channel weight ranges and makes per-tensor quantization harder.
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)      # (C_out,)

    folded = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
        conv.padding, conv.dilation, conv.groups, bias=True,
        device=conv.weight.device, dtype=conv.weight.dtype,
    )
    # reshape to (C_out, 1, 1, 1) so each output filter is multiplied by its own
    # scale and broadcasting handles the other three axes.
    folded.weight.data = conv.weight.data * s.reshape(-1, 1, 1, 1)
    # Our convs have bias=False, so prev_bias is zeros; the branch keeps the
    # function correct for a conv that does carry one.
    prev_bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    # b' = beta + s*(b - mu). With b = 0 this is beta - s*mu, the constant term
    # left over once the mean subtraction is pushed into the convolution.
    folded.bias.data = bn.bias.data + s * (prev_bias - bn.running_mean)
    return folded


@torch.no_grad()
def fold_model(model: nn.Module) -> Tuple[nn.Module, int]:
    """Fold every adjacent (Conv2d, BatchNorm2d) pair inside nn.Sequential blocks.

    MobileNet-v2 places every BatchNorm directly after its convolution inside a
    Sequential (`ConvBNReLU` is Conv/BN/ReLU6; each block's projection is
    Conv/BN at the tail of `InvertedResidual.conv`), so scanning Sequential
    children for adjacent pairs covers all 52 of them.

    The BatchNorm is replaced by nn.Identity rather than deleted, so module
    indices - and therefore the state-dict keys and the activation-quantizer
    hook sites - stay stable.

    Returns (model, number_of_folds). The model is modified in place; pass a
    copy if the original is still needed.
    """
    folds = 0
    # Only Sequential containers are scanned, because adjacency inside a
    # Sequential is what guarantees the BatchNorm consumes that conv's output
    # directly. Two modules being adjacent in named_modules() would prove
    # nothing about how data flows between them.
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        for i in range(len(module) - 1):
            conv, bn = module[i], module[i + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                module[i] = fold_conv_bn(conv, bn)
                # Identity rather than deletion: removing the entry would shift
                # every later index, breaking state-dict keys and the
                # activation-quantizer hook sites. Identity costs nothing at
                # run time and keeps ConvBNReLU matching isinstance checks.
                module[i + 1] = nn.Identity()
                folds += 1
    return model, folds


@torch.no_grad()
def verify_fold(original: nn.Module, folded: nn.Module, device: torch.device,
                shape: Tuple[int, ...] = (8, 3, 32, 32), tol: float = 1e-3) -> dict:
    """Check that folding preserved the function the network computes.

    Folding is only worth anything if it is exact, and the accounting in
    sizing.py charges nothing for BatchNorm once `fold_bn` is set. That makes
    this check the thing standing between a legitimate 4x reduction in
    BatchNorm storage and a silent under-report of model size.
    """
    original.eval().to(device)
    folded.eval().to(device)
    x = torch.randn(*shape, device=device)
    a, b = original(x), folded(x)
    diff = (a - b).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "logit_scale": a.abs().mean().item(),
        "agree": bool(diff.max().item() < tol),
        "same_argmax": bool((a.argmax(1) == b.argmax(1)).all().item()),
        "remaining_bn": sum(1 for m in folded.modules() if isinstance(m, nn.BatchNorm2d)),
    }

"""Linear (affine) quantization, written from scratch.

No quantization library is used anywhere in this file: the mapping, the range
calibration, the fake-quantize round trip and the straight-through gradient are
all implemented directly, as required by the assignment.

Two quantizer families are provided:

  * `LinearQuantizer`  - the arithmetic. Symmetric or asymmetric, per-tensor or
    per-output-channel, arbitrary bit width.
  * `ActivationQuantizer` - an nn.Module that observes activation ranges during
    a calibration pass and then fake-quantizes at inference.

Notation used throughout (b = bit width):

    q  = clamp(round(x / s) + z, q_min, q_max)      quantize
    x^ = s * (q - z)                                dequantize

  symmetric  : z = 0,  s = max|x| / (2^(b-1) - 1),  q in [-2^(b-1)+1, 2^(b-1)-1]
  asymmetric : s = (max - min) / (2^b - 1),
               z = round(-min / s),                 q in [0, 2^b - 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

QScheme = Literal["symmetric", "asymmetric"]
QGranularity = Literal["per_tensor", "per_channel"]


# --------------------------------------------------------------------------- #
# Core arithmetic
# --------------------------------------------------------------------------- #
def quant_bounds(num_bits: int, scheme: QScheme) -> Tuple[int, int]:
    """Integer range representable in `num_bits` under the given scheme.

    For the symmetric scheme we use the *restricted* range, i.e. -(2^(b-1) - 1)
    rather than -2^(b-1). Giving up the single extra negative code makes the
    range exactly symmetric about zero, which keeps the dequantized grid
    symmetric too. Weight distributions are close to zero-mean, so a lopsided
    grid biases every layer slightly; the cost is one unused code out of 2^b.
    """
    if scheme == "symmetric":
        qmax = 2 ** (num_bits - 1) - 1
        return -qmax, qmax
    return 0, 2 ** num_bits - 1


def compute_qparams(
    x: torch.Tensor,
    num_bits: int,
    scheme: QScheme = "symmetric",
    granularity: QGranularity = "per_tensor",
    channel_dim: int = 0,
    percentile: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive (scale, zero_point) for a tensor.

    Args:
        x: tensor whose range is being measured.
        num_bits: bit width b.
        scheme: 'symmetric' or 'asymmetric'.
        granularity: 'per_tensor' gives one scalar scale; 'per_channel' gives
            one scale per slice along `channel_dim`. Per-channel is essential
            for MobileNet weights: depthwise convolutions do no cross-channel
            mixing, so per-channel weight ranges diverge by orders of magnitude
            and a single shared scale quantizes the narrow channels to all-zero.
        channel_dim: which axis indexes output channels (0 for conv weights of
            shape (Cout, Cin/groups, kh, kw), 0 for Linear (out, in)).
        percentile: if given (e.g. 99.9), clip the observed range to this
            percentile of |x| instead of the true min/max. Trades a few large
            outliers for a finer grid over the bulk of the distribution.

    Returns:
        (scale, zero_point). Shapes are scalars for per_tensor, or 1-D of
        length Cout for per_channel. zero_point is 0 for the symmetric scheme.
    """
    qmin, qmax = quant_bounds(num_bits, scheme)

    if granularity == "per_channel":
        # Flatten every axis except the channel axis, so we can reduce over dim 1.
        perm = x.movedim(channel_dim, 0)
        flat = perm.reshape(perm.shape[0], -1)
        reduce_dim = 1
    else:
        flat = x.reshape(1, -1)
        reduce_dim = 1

    if scheme == "symmetric":
        if percentile is not None:
            absmax = torch.quantile(flat.abs().float(), percentile / 100.0, dim=reduce_dim)
        else:
            absmax = flat.abs().amax(dim=reduce_dim)
        # A channel of all zeros would give scale 0 and produce NaNs; clamp it.
        scale = (absmax / qmax).clamp(min=1e-12)
        zero_point = torch.zeros_like(scale)
    else:
        if percentile is not None:
            lo = torch.quantile(flat.float(), (100.0 - percentile) / 100.0, dim=reduce_dim)
            hi = torch.quantile(flat.float(), percentile / 100.0, dim=reduce_dim)
        else:
            lo = flat.amin(dim=reduce_dim)
            hi = flat.amax(dim=reduce_dim)
        # Always include 0 in the represented range, so that a genuine zero in
        # the tensor maps to an exact integer code (important after ReLU, where
        # a large fraction of the values are exactly 0).
        lo = torch.minimum(lo, torch.zeros_like(lo))
        hi = torch.maximum(hi, torch.zeros_like(hi))
        scale = ((hi - lo) / (qmax - qmin)).clamp(min=1e-12)
        zero_point = torch.round(qmin - lo / scale).clamp(qmin, qmax)

    if granularity == "per_tensor":
        scale, zero_point = scale.squeeze(), zero_point.squeeze()
    return scale, zero_point


def _broadcast_shape(x: torch.Tensor, param: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Reshape a per-channel (scale, zp) vector so it broadcasts against x."""
    if param.ndim == 0:
        return param
    shape = [1] * x.ndim
    shape[channel_dim] = -1
    return param.reshape(shape)


def quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
             num_bits: int, scheme: QScheme, channel_dim: int = 0) -> torch.Tensor:
    """Float tensor -> integer codes. Returns integers stored in a float tensor."""
    qmin, qmax = quant_bounds(num_bits, scheme)
    s = _broadcast_shape(x, scale, channel_dim)
    z = _broadcast_shape(x, zero_point, channel_dim)
    return torch.clamp(torch.round(x / s) + z, qmin, qmax)


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
               channel_dim: int = 0) -> torch.Tensor:
    """Integer codes -> float tensor on the quantization grid."""
    s = _broadcast_shape(q, scale, channel_dim)
    z = _broadcast_shape(q, zero_point, channel_dim)
    return (q - z) * s


class _RoundSTE(torch.autograd.Function):
    """Straight-through estimator for the non-differentiable round().

    round() has zero gradient almost everywhere, so a quantized network would
    receive no gradient at all. The STE substitutes the identity on the
    backward pass: d(round(x))/dx := 1. This is what makes quantization-aware
    training possible, and it is why QAT recovers accuracy that post-training
    quantization alone cannot.

    Gradients are masked outside the clamping range, so weights that have been
    pushed beyond the representable range are not encouraged to run further
    away (this is the 'clipped STE' variant).
    """

    @staticmethod
    def forward(ctx, x, qmin, qmax):
        ctx.save_for_backward(x)
        ctx.qmin, ctx.qmax = qmin, qmax
        return torch.clamp(torch.round(x), qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x >= ctx.qmin) & (x <= ctx.qmax)
        return grad_output * mask.to(grad_output.dtype), None, None


def fake_quantize(x: torch.Tensor, num_bits: int, scheme: QScheme = "symmetric",
                  granularity: QGranularity = "per_tensor", channel_dim: int = 0,
                  percentile: Optional[float] = None,
                  scale: Optional[torch.Tensor] = None,
                  zero_point: Optional[torch.Tensor] = None,
                  differentiable: bool = False) -> torch.Tensor:
    """Quantize then immediately dequantize: the value lands on the grid but
    stays a float tensor, so the rest of the network runs unchanged.

    This is the standard way to *simulate* quantization. The accuracy it
    reports is exactly the accuracy an integer kernel would produce, while
    letting us keep using ordinary float convolutions.

    Set `differentiable=True` during quantization-aware training so gradients
    flow through the straight-through estimator.
    """
    if scale is None or zero_point is None:
        scale, zero_point = compute_qparams(x, num_bits, scheme, granularity,
                                            channel_dim, percentile)
    qmin, qmax = quant_bounds(num_bits, scheme)
    s = _broadcast_shape(x, scale, channel_dim)
    z = _broadcast_shape(x, zero_point, channel_dim)

    if differentiable:
        q = _RoundSTE.apply(x / s + z, qmin, qmax)
    else:
        q = torch.clamp(torch.round(x / s + z), qmin, qmax)
    return (q - z) * s


# --------------------------------------------------------------------------- #
# Activation quantization
# --------------------------------------------------------------------------- #
@dataclass
class ActQuantConfig:
    """Configuration for a single activation quantization site."""

    num_bits: int = 8
    scheme: QScheme = "asymmetric"
    # Fraction of the observed range to keep. 99.99 discards only extreme
    # outliers; lowering it narrows the grid and can *improve* accuracy by
    # spending resolution where the data actually is.
    percentile: Optional[float] = 99.99
    # Exponential moving average factor used while calibrating over batches.
    ema_momentum: float = 0.1


class ActivationQuantizer(nn.Module):
    """Observes activation ranges during calibration, fake-quantizes afterwards.

    Lifecycle:
        1. `calibrating = True`  - forward passes only *record* running min/max;
           the tensor passes through untouched.
        2. `freeze()`            - converts the observed range into (scale, zp).
        3. `enabled = True`      - forward passes fake-quantize.

    Calibration uses an exponential moving average of per-batch min/max rather
    than a global max, so a single pathological batch cannot blow up the range
    for every subsequent inference.

    Why the scheme differs by site: tensors that follow ReLU6 live in [0, 6] and
    are one-sided, so an asymmetric quantizer spends all 2^b codes on the range
    that actually occurs. Tensors that follow a linear-bottleneck projection are
    signed and roughly zero-mean, so a symmetric quantizer is the right fit.
    `pipeline.py` chooses per site on this basis.
    """

    def __init__(self, cfg: ActQuantConfig, name: str = "") -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.calibrating = False
        self.enabled = False
        self.frozen = False

        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        # Number of elements seen, used for the activation-size accounting (Q4b).
        self.register_buffer("numel_per_sample", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _observe(self, x: torch.Tensor) -> None:
        """Update the running range from one batch."""
        if self.cfg.percentile is not None and x.numel() > 1:
            # torch.quantile has a size limit; subsample large tensors.
            flat = x.detach().flatten().float()
            if flat.numel() > 1_000_000:
                idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
                flat = flat[idx]
            p = self.cfg.percentile / 100.0
            batch_max = torch.quantile(flat, p)
            batch_min = torch.quantile(flat, 1.0 - p)
        else:
            batch_min, batch_max = x.detach().amin(), x.detach().amax()

        if torch.isinf(self.running_min):
            self.running_min.fill_(batch_min.item())
            self.running_max.fill_(batch_max.item())
        else:
            m = self.cfg.ema_momentum
            self.running_min.mul_(1 - m).add_(m * batch_min)
            self.running_max.mul_(1 - m).add_(m * batch_max)

        if self.numel_per_sample.item() == 0:
            self.numel_per_sample.fill_(x[0].numel())

    @torch.no_grad()
    def freeze(self) -> None:
        """Turn the observed range into concrete (scale, zero_point)."""
        qmin, qmax = quant_bounds(self.cfg.num_bits, self.cfg.scheme)
        lo, hi = self.running_min.clone(), self.running_max.clone()

        if self.cfg.scheme == "symmetric":
            absmax = torch.maximum(lo.abs(), hi.abs()).clamp(min=1e-12)
            self.scale.fill_((absmax / qmax).item())
            self.zero_point.fill_(0.0)
        else:
            lo = torch.minimum(lo, torch.zeros_like(lo))
            hi = torch.maximum(hi, torch.zeros_like(hi))
            scale = ((hi - lo) / (qmax - qmin)).clamp(min=1e-12)
            self.scale.fill_(scale.item())
            self.zero_point.fill_(torch.round(qmin - lo / scale).clamp(qmin, qmax).item())

        self.frozen = True
        self.calibrating = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self._observe(x)
            return x
        if not self.enabled:
            return x
        return fake_quantize(x, self.cfg.num_bits, self.cfg.scheme,
                             granularity="per_tensor",
                             scale=self.scale, zero_point=self.zero_point,
                             differentiable=self.training)

    def extra_repr(self) -> str:
        return (f"name={self.name}, bits={self.cfg.num_bits}, scheme={self.cfg.scheme}, "
                f"range=[{self.running_min.item():.3f}, {self.running_max.item():.3f}], "
                f"enabled={self.enabled}")

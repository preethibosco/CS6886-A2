"""MobileNet-v2, written from scratch and adapted for 32x32 CIFAR-10 inputs.

We implement the architecture ourselves rather than importing
`torchvision.models.mobilenet_v2` for two reasons:
  1. Q1(b) asks us to describe our own configuration.
  2. The activation quantizer (src/compress/quantize.py) needs to attach at
     specific points inside each inverted residual block. Owning the module
     structure makes those attachment points explicit and named.

Reference architecture: Sandler et al., "MobileNetV2: Inverted Residuals and
Linear Bottlenecks", CVPR 2018.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def _make_divisible(value: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Round a channel count to the nearest multiple of `divisor`.

    Channel counts scaled by the width multiplier are rounded to multiples of 8
    so that the resulting tensors stay friendly to vectorised kernels. The
    guard below prevents the rounding from ever removing more than 10% of the
    channels, which would silently change the architecture at small widths.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBNReLU(nn.Sequential):
    """conv -> BatchNorm -> ReLU6.

    Bias is disabled on the convolution because the following BatchNorm has its
    own learnable shift, making the conv bias redundant (and, at inference, it
    would be an extra tensor to store and quantize for no benefit).

    ReLU6 (clamp to [0, 6]) rather than plain ReLU: the fixed upper bound keeps
    activations in a known range under low-precision arithmetic, which is
    exactly the property the activation quantizer exploits later.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """The MobileNet-v2 inverted residual block.

        expand (1x1, t*C)  ->  depthwise (3x3, stride s)  ->  project (1x1, C')

    The expansion layer is skipped entirely when the expansion ratio is 1 (the
    very first block), since a 1x1 conv from C to C would be a no-op layer.

    The projection convolution is deliberately *linear* - BatchNorm but no
    activation. This is the "linear bottleneck": applying ReLU in the narrow
    output space would irrecoverably destroy information. Downstream this means
    projection outputs are signed and roughly zero-mean, so they call for a
    symmetric activation quantizer, whereas post-ReLU6 tensors are one-sided
    and call for an asymmetric one.

    The residual connection is only valid when the block preserves shape, i.e.
    stride 1 and in_channels == out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        assert stride in (1, 2), f"stride must be 1 or 2, got {stride}"
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        layers: List[nn.Module] = []

        if expand_ratio != 1:
            # Point-wise expansion C -> t*C.
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            # Depth-wise 3x3: groups == channels means one filter per channel.
            # This is where the spatial mixing happens, and it is cheap because
            # it does no cross-channel mixing.
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            # Point-wise projection t*C -> C', linear (no ReLU, see docstring).
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNet-v2 for CIFAR-10.

    Architecture table, given as (t, c, n, s):
        t = expansion ratio, c = output channels, n = repeats, s = stride of
        the first repeat (the remaining n-1 repeats always use stride 1).

    CIFAR adaptation
    ----------------
    The ImageNet configuration downsamples 32x (224 -> 7). Applied to a 32x32
    input that yields a 1x1 feature map by the c=64 stage, and top-1 stalls
    near 88%. We therefore remove two downsampling steps:
        * stem convolution stride 2 -> 1
        * the c=24 stage stride 2 -> 1
    Total downsampling becomes 8x, so a 32x32 input reaches a 4x4 final feature
    map - the same spatial budget the ImageNet model enjoys, reached from a
    smaller input. Everything else is unchanged from the paper.

    Args:
        num_classes: number of output logits (10 for CIFAR-10).
        width_mult: uniform channel scaling. 1.0 gives ~2.24M parameters.
        dropout: dropout probability applied to the pooled feature vector just
            before the classifier. MobileNet-v2 has no other regularisation
            layer, so this is the main one besides weight decay.
        bn_momentum: BatchNorm running-statistic momentum. The torch default of
            0.1 is noisy at batch size 128 on 50k images; 0.05 gives smoother
            running estimates, which matters because those frozen statistics
            are what the quantized model runs with at evaluation time.
        cifar_stem: if True apply the stride adaptation described above. Set to
            False to recover the stock ImageNet topology.
    """

    # (expand_ratio t, output channels c, repeats n, stride s)
    INVERTED_RESIDUAL_SETTING = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        bn_momentum: float = 0.05,
        cifar_stem: bool = True,
        round_nearest: int = 8,
    ) -> None:
        super().__init__()
        self.width_mult = width_mult
        self.cifar_stem = cifar_stem

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        # The final 1x1 expansion is never scaled *down* below 1280: shrinking
        # it hurts accuracy much more than it saves parameters.
        last_channel = _make_divisible(1280 * max(1.0, width_mult), round_nearest)
        self.last_channel = last_channel

        stem_stride = 1 if cifar_stem else 2
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, kernel_size=3, stride=stem_stride)]

        for t, c, n, s in self.INVERTED_RESIDUAL_SETTING:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                # Only the first repeat of a stage carries the stride.
                stride = s if i == 0 else 1
                # CIFAR adaptation: cancel the c=24 stage's downsample.
                if cifar_stem and c == 24 and i == 0:
                    stride = 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        self._initialise_weights(bn_momentum)

    def _initialise_weights(self, bn_momentum: float) -> None:
        """Kaiming-normal for convolutions, unit-scale BatchNorm, small-normal head.

        Kaiming ('fan_out', relu) is the right initialiser here because every
        convolution is followed by BatchNorm and a ReLU6; it keeps the variance
        of the pre-activations stable through the 53 conv layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.momentum = bn_momentum
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v2_cifar(num_classes: int = 10, width_mult: float = 1.0, dropout: float = 0.2,
                       bn_momentum: float = 0.05) -> MobileNetV2:
    """Convenience constructor for the CIFAR-10 configuration used in this report."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult,
                       dropout=dropout, bn_momentum=bn_momentum, cifar_stem=True)

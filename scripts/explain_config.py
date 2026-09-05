"""Unroll the MobileNet-v2 (t, c, n, s) table into concrete per-block channels.

Shows exactly where every hidden ("expanded") channel count comes from, and
counts the weight-bearing layers that give MobileNet-v2 its quoted depth.

Run:  python scripts/explain_config.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mobilenetv2 import MobileNetV2, _make_divisible

WIDTH = 1.0
stem_out = _make_divisible(32 * WIDTH)

print("Config table rows are (t, c, n, s):")
print("  t = expansion ratio   c = OUTPUT channels of the block")
print("  n = how many times the block repeats   s = stride of the FIRST repeat\n")

hdr = (f"{'idx':<6}{'stage(t,c,n,s)':<18}{'Cin':>5}{'  x t  ':^7}{'hidden':>7}"
       f"{'  ->  ':^6}{'Cout':>5}{'  s':>4}{'   residual?':<13}{'convs':>6}")
print(hdr)
print("-" * len(hdr))

cin = stem_out
idx = 1
layers = 1          # the stem convolution
block_count = 0
print(f"{'0':<6}{'stem conv 3x3':<18}{3:>5}{'':^7}{'-':>7}{'':^6}{stem_out:>5}{1:>4}{'':<13}{1:>6}")

for (t, c, n, s) in MobileNetV2.INVERTED_RESIDUAL_SETTING:
    cout = _make_divisible(c * WIDTH)
    for i in range(n):
        stride = s if i == 0 else 1
        if c == 24 and i == 0:          # CIFAR adaptation
            stride = 1
        hidden = int(round(cin * t))    # <-- hidden depends on Cin, NOT Cout
        # t == 1 means the expansion 1x1 conv is omitted entirely.
        convs = 2 if t == 1 else 3
        residual = "yes (x + f(x))" if (stride == 1 and cin == cout) else "no"
        print(f"{idx:<6}{f'({t},{c},{n},{s})':<18}{cin:>5}{f'x{t}':^7}{hidden:>7}"
              f"{'-->':^6}{cout:>5}{stride:>4}   {residual:<13}{convs:>6}")
        layers += convs
        block_count += 1
        cin = cout
        idx += 1

head_out = _make_divisible(1280 * max(1.0, WIDTH))
print(f"{idx:<6}{'head conv 1x1':<18}{cin:>5}{'':^7}{'-':>7}{'':^6}{head_out:>5}{1:>4}{'':<13}{1:>6}")
layers += 1
print(f"{'':<6}{'classifier linear':<18}{head_out:>5}{'':^7}{'-':>7}{'':^6}{10:>5}{'':>4}{'':<13}{1:>6}")
layers += 1

print("-" * len(hdr))
print(f"\ninverted residual blocks : {block_count}   (sum of n = "
      f"{'+'.join(str(r[2]) for r in MobileNetV2.INVERTED_RESIDUAL_SETTING)} = {block_count})")
print(f"weight-bearing layers    : {layers}")
print("\n  depth breakdown:")
print("    stem conv                                    1")
print("    features.1  (t=1, expansion omitted)  2 convs 2")
print("    features.2..17  (16 blocks x 3 convs)        48")
print("    head 1x1 conv                                1")
print("    classifier linear                            1")
print("    " + "-" * 45)
print(f"    total                                       {layers}")

# Cross-check against the real module tree.
import torch.nn as nn
m = MobileNetV2(num_classes=10, width_mult=WIDTH, cifar_stem=True)
real = sum(1 for _ in m.modules() if isinstance(_, (nn.Conv2d, nn.Linear)))
real_blocks = sum(1 for mod in m.features if type(mod).__name__ == "InvertedResidual")
print(f"\n  cross-check against built model: {real} weight layers, {real_blocks} blocks "
      f"-> {'MATCH' if (real, real_blocks) == (layers, block_count) else 'MISMATCH'}")

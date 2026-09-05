"""Sanity-check the CIFAR MobileNet-v2: shape progression and parameter budget.

Run:  python scripts/inspect_model.py
Prints the spatial resolution after every stage so the CIFAR stride adaptation
can be verified by eye, plus where the parameters actually live (which is what
the compression pipeline has to target).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.models.mobilenetv2 import mobilenet_v2_cifar, MobileNetV2
from src.utils import count_parameters

model = mobilenet_v2_cifar()
model.eval()

print("=== spatial progression (input 1x3x32x32) ===")
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    for i, block in enumerate(model.features):
        x = block(x)
        kind = type(block).__name__
        print(f"  features[{i:2d}] {kind:<17s} -> {tuple(x.shape)}")
    x = model.pool(x)
    logits = model.classifier(torch.flatten(x, 1))
print(f"  logits -> {tuple(logits.shape)}")

print("\n=== parameter budget ===")
total = count_parameters(model)
print(f"  total parameters : {total:,}")

# Where do parameters live? This drives what the compression pipeline targets.
buckets = {"pointwise 1x1 conv": 0, "depthwise 3x3 conv": 0, "other conv": 0,
           "batchnorm": 0, "classifier": 0}
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        n = m.weight.numel()
        if m.groups > 1:
            buckets["depthwise 3x3 conv"] += n
        elif m.kernel_size == (1, 1):
            buckets["pointwise 1x1 conv"] += n
        else:
            buckets["other conv"] += n
    elif isinstance(m, nn.BatchNorm2d):
        buckets["batchnorm"] += m.weight.numel() + m.bias.numel()
    elif isinstance(m, nn.Linear):
        buckets["classifier"] += m.weight.numel() + m.bias.numel()

for k, v in buckets.items():
    print(f"  {k:<20s} {v:>10,}  ({100*v/total:5.2f}%)")
print(f"  {'SUM':<20s} {sum(buckets.values()):>10,}")

print(f"\n  fp32 size (params only)      : {total*4/2**20:.3f} MB")
bn_buffers = sum(b.numel() for n, b in model.named_buffers() if 'running_' in n)
print(f"  BN running stats (buffers)   : {bn_buffers:,} values = {bn_buffers*4/2**20:.3f} MB")

print("\n=== ImageNet-stem variant, for contrast ===")
stock = MobileNetV2(num_classes=10, cifar_stem=False).eval()
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    for i, block in enumerate(stock.features):
        x = block(x)
print(f"  final feature map with stock strides on a 32x32 input: {tuple(x.shape)}")

"""Full layer-by-layer breakdown of the CIFAR MobileNet-v2.

Emits, for every convolution / BN / linear layer:
  - its role (stem / expand / depthwise / project / head / classifier)
  - shape parameters (Cin, Cout, kernel, stride, groups)
  - the weight-parameter count, with the formula that produced it
  - the output activation tensor size (elements, for batch size 1)

The activation column is what the activation-compression analysis (Q4b) is
measured against, so it is produced here rather than being re-derived later.

Run:  python scripts/layer_table.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.models.mobilenetv2 import mobilenet_v2_cifar

model = mobilenet_v2_cifar().eval()

# ---------------------------------------------------------------- hook capture
# Record the output shape of every leaf conv / bn / linear by running one image.
shapes = {}
handles = []


def _hook(name):
    def fn(mod, inp, out):
        shapes[name] = tuple(out.shape)
    return fn


for name, m in model.named_modules():
    if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
        handles.append(m.register_forward_hook(_hook(name)))

with torch.no_grad():
    model(torch.randn(1, 3, 32, 32))
for h in handles:
    h.remove()


def role_of(name, mod):
    """Classify a conv by its position inside the inverted-residual block."""
    if not isinstance(mod, nn.Conv2d):
        return ""
    if name == "features.0.0":
        return "stem 3x3"
    if name == "features.18.0":
        return "head 1x1"
    if mod.groups > 1:
        return "depthwise 3x3"
    # Inside a block, conv index 0 is the expansion, the last conv is the projection.
    return "expand 1x1" if name.endswith(".conv.0.0") else "project 1x1"


rows = []
tot_w = tot_bn = tot_buf = 0
peak_act = 0

for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        cout, cin_g, kh, kw = m.weight.shape
        n = m.weight.numel()
        if m.groups > 1:
            formula = f"{kh}x{kw}x{cout}"           # depthwise: one kxk filter per channel
        elif kh == 1:
            formula = f"{m.in_channels}x{cout}"     # pointwise
        else:
            formula = f"{kh}x{kw}x{m.in_channels}x{cout}"
        act = shapes.get(name, ())
        act_n = int(torch.tensor(act).prod()) if act else 0
        peak_act = max(peak_act, act_n)
        tot_w += n
        rows.append((name, role_of(name, m), f"{m.in_channels}->{m.out_channels}",
                     f"{kh}x{kw}", m.stride[0], m.groups, formula, n,
                     f"{act[1]}x{act[2]}x{act[3]}" if len(act) == 4 else "-", act_n))
    elif isinstance(m, nn.BatchNorm2d):
        tot_bn += 2 * m.num_features
        tot_buf += 2 * m.num_features
    elif isinstance(m, nn.Linear):
        n = m.weight.numel() + m.bias.numel()
        tot_w += n
        rows.append((name, "classifier", f"{m.in_features}->{m.out_features}", "-", "-", "-",
                     f"{m.in_features}x{m.out_features}+{m.out_features}", n, "-", m.out_features))

hdr = f"{'layer':<22}{'role':<15}{'ch':<12}{'k':<5}{'s':<3}{'grp':<6}{'weights formula':<20}{'#w':>10}  {'out act':<14}{'act#':>8}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:<22}{r[1]:<15}{r[2]:<12}{r[3]:<5}{str(r[4]):<3}{str(r[5]):<6}{r[6]:<20}{r[7]:>10,}  {r[8]:<14}{r[9]:>8,}")

print("-" * len(hdr))
print(f"{'conv+linear weights':<40}{tot_w:>10,}")
print(f"{'batchnorm gamma+beta (learnable)':<40}{tot_bn:>10,}")
print(f"{'TOTAL PARAMETERS':<40}{tot_w + tot_bn:>10,}")
print(f"{'batchnorm running_mean/var (buffers)':<40}{tot_buf:>10,}   <- needed at inference, NOT in parameters()")
print(f"{'inference-time storage total':<40}{tot_w + tot_bn + tot_buf:>10,}")
print()
print(f"fp32 parameters          : {(tot_w+tot_bn)*4/2**20:.3f} MB")
print(f"fp32 params + BN buffers : {(tot_w+tot_bn+tot_buf)*4/2**20:.3f} MB")
print(f"depth (conv + linear layers with weights) : {len(rows)}")
print(f"peak single activation tensor (batch=1)   : {peak_act:,} elements = {peak_act*4/2**20:.3f} MB fp32")
print(f"sum of all conv output activations (b=1)  : {sum(r[9] for r in rows):,} elements = {sum(r[9] for r in rows)*4/2**20:.3f} MB fp32")

# --------------------------------------------- depthwise-separable saving check
print("\n--- why depthwise separable: worked example, block features.3 (24ch, t=6) ---")
dw = 3 * 3 * 144
pw = 144 * 24
std = 3 * 3 * 144 * 24
print(f"  depthwise 3x3 on 144ch      : 3*3*144            = {dw:,}")
print(f"  pointwise 1x1 144->24       : 144*24             = {pw:,}")
print(f"  separable total             :                      {dw+pw:,}")
print(f"  one standard 3x3 144->24    : 3*3*144*24         = {std:,}")
print(f"  saving                      : {std/(dw+pw):.2f}x fewer weights")

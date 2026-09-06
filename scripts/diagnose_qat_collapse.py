"""Why does quantization-aware fine-tuning sometimes collapse irrecoverably?

Symptom: at w3 / sparsity 0.7 a fine-tuning run went to 10% top-1 at epoch 1 and
never left, while the identical configuration succeeded in a separate run
(87.5% by epoch 3). A failure that depends on the RNG stream rather than the
configuration points at an instability, not a bug.

Hypothesis: per-tensor quantization derives its scale from max|w| over the whole
tensor. If one master weight grows during fine-tuning, the scale grows with it,
and every other weight in that tensor rounds to zero. The layer dies, its
gradient vanishes, and no subsequent step can revive it - the failure is
absorbing.

This script instruments the first epoch: gradient norm, max|w|, and the fraction
of live weights that the current scale sends to code 0.

Run:  python scripts/diagnose_qat_collapse.py
"""
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.compress.pipeline import (CompressionConfig, attach_activation_quantizers,
                                   calibrate_activations, compress)
from src.compress.prune import PruneConfig, Pruner
from src.compress.qat import _layer_policy, _quantize_in_place, _restore
from src.data import DataConfig, build_loaders
from src.models.mobilenetv2 import mobilenet_v2_cifar
from src.utils import get_device, seed_everything

seed_everything(42)
dev = get_device()
train_loader, test_loader, calib = build_loaders(DataConfig(seed=42), download=False)
base = mobilenet_v2_cifar().to(dev)
base.load_state_dict(torch.load("checkpoints/mobilenetv2_cifar10_best.pt",
                                map_location=dev, weights_only=False)["model"])

cfg = CompressionConfig(weight_bits=3, activation_bits=8, depthwise_bits=8,
                        edge_bits=8, bn_bits=8, sparsity=0.7,
                        weight_method="linear_per_tensor")
# Replicate run_qat.py exactly, including the PTQ pass that consumes RNG first.
_ = compress(base, cfg, calib, dev)

work = copy.deepcopy(base).to(dev)
policy = _layer_policy(work, cfg)
pr = Pruner(work, PruneConfig(sparsity=0.7, scope="global", max_layer_sparsity=0.95))
pr.compute_masks(); pr.apply()
masks = pr.masks
qs, _ = attach_activation_quantizers(work, cfg)
calibrate_activations(work, qs, calib, dev)

crit = nn.CrossEntropyLoss(label_smoothing=0.1)
opt = torch.optim.SGD(work.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)


def probe():
    mx, worst, dead = 0.0, "", []
    for n, p in policy.items():
        w = p["module"].weight.data
        a = w.abs().max().item()
        if a > mx:
            mx, worst = a, n
        if p["bits"] == 3:
            s = a / 3.0                      # per-tensor step size
            m = masks.get(n, torch.ones_like(w, dtype=torch.bool))
            dead.append(((w.abs() < s / 2) & m).float().mean().item())
    return mx, worst, 100 * sum(dead) / len(dead)


mx, worst, dead = probe()
print(f"  {'start':<10} max|w|={mx:.4f} ({worst})   live weights -> code 0: {dead:.1f}%")

work.train()
for it, (x, y) in enumerate(train_loader):
    x, y = x.to(dev), y.to(dev)
    m = _quantize_in_place(policy, masks, cfg, None)
    opt.zero_grad(set_to_none=True)
    loss = crit(work(x), y)
    loss.backward()
    gn = math.sqrt(sum(float((p.grad ** 2).sum()) for p in work.parameters()
                       if p.grad is not None))
    _restore(policy, m)
    opt.step()
    pr.apply()
    if it in (0, 1, 2, 5, 10, 25, 50, 100, 200, 390):
        mx, worst, dead = probe()
        print(f"  iter {it:<5} loss={loss.item():6.3f} |grad|={gn:8.2f}  "
              f"max|w|={mx:7.4f} ({worst[:28]})  -> code 0: {dead:5.1f}%", flush=True)
    if it >= 390:
        break

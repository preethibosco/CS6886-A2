"""Measure how the per-layer method-selection tolerance trades ratio against accuracy.

The selector admits any quantization method whose reconstruction MSE is within
`tolerance` of the best method for that layer, then takes the cheapest in bits.
tolerance = 1.0 is "always pick the lowest-distortion method"; a large tolerance
is "always pick the cheapest method". Neither extreme is obviously right, so the
setting is chosen here by measurement.

Run:  python scripts/sweep_method_tolerance.py --checkpoint <path>
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.compress.pipeline import CompressionConfig, compress
from src.data import DataConfig, build_loaders
from src.evaluate import evaluate
from src.models.mobilenetv2 import mobilenet_v2_cifar
from src.utils import get_device, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
ap.add_argument("--weight-bits", type=int, default=4)
args = ap.parse_args()

seed_everything(42, deterministic=True)
device = get_device()
_, test_loader, calib_loader = build_loaders(DataConfig(seed=42), download=False)
model = mobilenet_v2_cifar().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                 weights_only=False)["model"])
base = evaluate(model, test_loader, device)
print(f"fp32 baseline top-1 = {base['top1']:.2f}%\n")

hdr = (f"{'setting':<22}{'model x':>9}{'weight x':>10}{'top-1':>8}{'drop':>8}"
       f"{'  methods chosen (by layer)':<44}")
print(hdr); print("-" * len(hdr))

settings = [("forced kmeans", dict(weight_method="kmeans")),
            ("forced lin/channel", dict(weight_method="linear_per_channel")),
            ("forced lin/tensor", dict(weight_method="linear_per_tensor"))]
settings += [(f"auto tol={t}", dict(weight_method="auto", method_mse_tolerance=t))
             for t in (1.0, 1.25, 2.0, 5.0, 20.0, 1e9)]

for label, kw in settings:
    cfg = CompressionConfig(weight_bits=args.weight_bits, activation_bits=8,
                            depthwise_bits=8, edge_bits=8, bn_bits=8, sparsity=0.0,
                            use_huffman=True, **kw)
    cmodel, res = compress(model, cfg, calib_loader, device)
    acc = evaluate(cmodel, test_loader, device)
    s = res.summary
    picks = Counter(l.notes.split(",")[0] for l in res.model_cost.layers
                    if l.kind in ("pointwise", "depthwise", "stem", "classifier") and l.notes)
    picks_s = " ".join(f"{k.replace('linear_','lin/').replace('per_','')}:{v}"
                       for k, v in sorted(picks.items()))
    print(f"{label:<22}{s['model_compression_ratio']:>9.2f}"
          f"{s['weight_compression_ratio']:>10.2f}{acc['top1']:>8.2f}"
          f"{acc['top1']-base['top1']:>8.2f}  {picks_s:<44}")

"""Sparsity x method sweep (post-training, no fine-tuning).

Pruning is the largest single lever in the pipeline and interacts with both
later stages: it changes which sparse layout wins, and it removes the small
weights that made the linear/tensor code distribution so compressible in the
first place. So it has to be measured jointly with the quantization method
rather than tuned on its own.

Run:  python scripts/sweep_sparsity.py --checkpoint <path>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from collections import Counter
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
print(f"fp32 baseline top-1 = {base['top1']:.2f}%   (w{args.weight_bits} a8, PTQ only)\n")

hdr = (f"{'method':<16}{'sparsity':>9}{'model x':>9}{'weight x':>10}{'MB':>8}"
       f"{'top-1':>8}{'drop':>8}{'  encodings':<28}")
print(hdr); print("-" * len(hdr))

for method in ("kmeans", "linear_per_tensor"):
    for sp in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9):
        cfg = CompressionConfig(weight_bits=args.weight_bits, activation_bits=8,
                                depthwise_bits=8, edge_bits=8, bn_bits=8,
                                sparsity=sp, weight_method=method, use_huffman=True)
        cmodel, res = compress(model, cfg, calib_loader, device)
        acc = evaluate(cmodel, test_loader, device)
        s = res.summary
        encs = Counter(l.encoding for l in res.model_cost.layers if l.kind == "pointwise")
        enc_s = " ".join(f"{k}:{v}" for k, v in sorted(encs.items()))
        print(f"{method:<16}{sp:>9.2f}{s['model_compression_ratio']:>9.2f}"
              f"{s['weight_compression_ratio']:>10.2f}{s['compressed_mb']:>8.3f}"
              f"{acc['top1']:>8.2f}{acc['top1']-base['top1']:>8.2f}  {enc_s:<28}", flush=True)
    print()

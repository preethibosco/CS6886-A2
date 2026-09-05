"""End-to-end sweep of the entropy penalty lambda (rate-distortion operating point).

Per-layer MSE and bit counts do not determine top-1 accuracy, so the operating
point is chosen here by running the whole pipeline and measuring both axes the
assignment grades: compression ratio and accuracy.

Run:  python scripts/sweep_lambda.py --checkpoint <path>
"""
import argparse, os, sys
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
ap.add_argument("--sparsity", type=float, default=0.0)
args = ap.parse_args()

seed_everything(42, deterministic=True)
device = get_device()
_, test_loader, calib_loader = build_loaders(DataConfig(seed=42), download=False)
model = mobilenet_v2_cifar().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                 weights_only=False)["model"])
base = evaluate(model, test_loader, device)
print(f"fp32 baseline top-1 = {base['top1']:.2f}%   "
      f"(w{args.weight_bits}, sparsity {args.sparsity}, PTQ only, no fine-tuning)\n")

hdr = (f"{'method':<24}{'model x':>9}{'weight x':>10}{'MB':>8}{'top-1':>8}{'drop':>8}")
print(hdr); print("-" * len(hdr))

runs = [("kmeans (lam=0)", dict(weight_method="kmeans"))]
runs += [("linear/channel", dict(weight_method="linear_per_channel")),
         ("linear/tensor", dict(weight_method="linear_per_tensor"))]
runs += [(f"ECSQ lam={l:g}", dict(weight_method="ecsq", ecsq_lambda=l))
         for l in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)]
runs += [(f"auto+ECSQ lam={l:g}", dict(weight_method="auto", ecsq_lambda=l))
         for l in (3e-3, 1e-2)]

for label, kw in runs:
    cfg = CompressionConfig(weight_bits=args.weight_bits, activation_bits=8,
                            depthwise_bits=8, edge_bits=8, bn_bits=8,
                            sparsity=args.sparsity, use_huffman=True, **kw)
    cmodel, res = compress(model, cfg, calib_loader, device)
    acc = evaluate(cmodel, test_loader, device)
    s = res.summary
    print(f"{label:<24}{s['model_compression_ratio']:>9.2f}"
          f"{s['weight_compression_ratio']:>10.2f}{s['compressed_mb']:>8.3f}"
          f"{acc['top1']:>8.2f}{acc['top1']-base['top1']:>8.2f}", flush=True)

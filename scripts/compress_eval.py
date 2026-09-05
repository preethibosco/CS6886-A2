"""Run the compression pipeline at one configuration and report everything.

Run:  python scripts/compress_eval.py --checkpoint checkpoints/mobilenetv2_cifar10_best.pt \
          --weight-bits 4 --activation-bits 8 --sparsity 0.8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.compress.pipeline import CompressionConfig, compress
from src.data import DataConfig, build_loaders
from src.evaluate import evaluate
from src.models.mobilenetv2 import mobilenet_v2_cifar
from src.utils import get_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
    ap.add_argument("--weight-bits", type=int, default=4)
    ap.add_argument("--activation-bits", type=int, default=8)
    ap.add_argument("--depthwise-bits", type=int, default=8)
    ap.add_argument("--edge-bits", type=int, default=8)
    ap.add_argument("--bn-bits", type=int, default=8)
    ap.add_argument("--sparsity", type=float, default=0.0)
    ap.add_argument("--weight-method", default="auto",
                    choices=["auto", "kmeans", "linear_per_channel", "linear_per_tensor"])
    ap.add_argument("--no-huffman", action="store_true")
    ap.add_argument("--no-act-quant", action="store_true")
    ap.add_argument("--method-mse-tolerance", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-table", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = get_device()

    _, test_loader, calib_loader = build_loaders(DataConfig(seed=args.seed), download=False)

    model = mobilenet_v2_cifar().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    base = evaluate(model, test_loader, device)
    print(f"baseline (fp32)  top1={base['top1']:.2f}%  top5={base['top5']:.2f}%  "
          f"[checkpoint epoch {ckpt.get('epoch')}]\n")

    cfg = CompressionConfig(
        weight_bits=args.weight_bits, activation_bits=args.activation_bits,
        depthwise_bits=args.depthwise_bits, edge_bits=args.edge_bits,
        bn_bits=args.bn_bits, sparsity=args.sparsity,
        weight_method=args.weight_method, use_huffman=not args.no_huffman,
        quantize_activations=not args.no_act_quant,
        method_mse_tolerance=args.method_mse_tolerance,
    )
    print(f"config: {cfg.describe()}\n")

    cmodel, res = compress(model, cfg, calib_loader, device)
    acc = evaluate(cmodel, test_loader, device)

    if args.full_table:
        print(res.model_cost.table());  print()
    print(res.activation_cost.table() if res.activation_cost else "(activations not quantized)")

    s = res.summary
    print(f"\n{'='*74}\n RESULTS  ({cfg.describe()})\n{'='*74}")
    print(f"  accuracy  fp32 {base['top1']:.2f}%  ->  compressed {acc['top1']:.2f}%   "
          f"(drop {base['top1']-acc['top1']:+.2f} pts)")
    print(f"  model     {s['baseline_fp32_mb']:.3f} MB -> {s['compressed_mb']:.3f} MB   "
          f"ratio {s['model_compression_ratio']:.2f}x")
    print(f"  weights   ratio {s['weight_compression_ratio']:.2f}x  "
          f"({s['weights_compressed_mb']:.3f} MB)")
    if "activation_compression_ratio" in s:
        print(f"  activations ratio {s['activation_compression_ratio']:.2f}x  "
              f"(traffic {s['activation_total_fp32_mb']:.3f} -> "
              f"{s['activation_total_quant_mb']:.3f} MB/inference)")
    ov = res.model_cost.overhead_breakdown()
    print(f"\n  storage split: values {100*ov['value_share']:.1f}%  "
          f"index {100*ov['index_share']:.1f}%  metadata {100*ov['metadata_share']:.1f}%  "
          f"-> overhead {100*ov['overhead_share']:.1f}%")
    print("\n  by layer kind:")
    for kind, d in sorted(res.model_cost.by_kind().items(),
                          key=lambda kv: -kv[1]["total_bits"]):
        print(f"    {kind:<12} {d['numel']:>9,} values  {d['total_bits']/8/1024:>9.1f} KB  "
              f"{d['ratio']:>6.2f}x  {100*d['share_of_compressed']:>5.1f}% of compressed")


if __name__ == "__main__":
    main()

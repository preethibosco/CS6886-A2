"""Compression sweep with Weights & Biases logging (Assignment Q3).

Runs the compression pipeline across a grid of bit widths and sparsity levels,
logging one wandb run per configuration. The Parallel Coordinates chart that Q3
asks for is built from these runs: each line is one configuration, and the axes
are the knobs (weight bits, activation bits, sparsity) together with the
outcomes (compression ratio, model size, accuracy).

Usage:
    python -m src.sweep --checkpoint checkpoints/mobilenetv2_cifar10_best.pt
    python -m src.sweep --quick                 # small grid, for a smoke test
    python -m src.sweep --qat --qat-epochs 10   # fine-tune each configuration

Post-training quantization is used for the sweep because it makes a dense grid
affordable; the selected configuration is then fine-tuned (--qat) to produce the
headline number reported for Q4.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Dict, List

import torch

from .compress.pipeline import CompressionConfig, compress
from .compress.qat import QATConfig, finetune
from .data import DataConfig, build_loaders
from .evaluate import evaluate
from .models.mobilenetv2 import mobilenet_v2_cifar
from .utils import get_device, seed_everything


def run_one(model, cfg: CompressionConfig, loaders, device, base_top1: float,
            qat: bool = False, qcfg: QATConfig | None = None) -> Dict:
    """Compress (optionally fine-tune) at one configuration and measure everything."""
    train_loader, test_loader, calib_loader = loaders
    t0 = time.time()

    if qat:
        # Order matters. Fine-tuning changes the weights, and changed weights
        # compress differently: training with the quantization grid in the loop
        # concentrates weights onto fewer codes, which lowers the code entropy
        # and makes the Huffman stage more effective. Measuring size before
        # fine-tuning would describe a model nobody evaluates.
        tuned, hist = finetune(model, cfg, qcfg, train_loader, test_loader,
                               calib_loader, device)
        # baseline_model=model, not `tuned`: the ratio must be quoted against
        # the original fp32 network. `tuned` may already have BatchNorm folded
        # away, and measuring against that understates the ratio.
        cmodel, res = compress(tuned, cfg, calib_loader, device, baseline_model=model)
    else:
        cmodel, res = compress(model, cfg, calib_loader, device)
        hist = []

    acc = evaluate(cmodel, test_loader, device)
    s = res.summary
    ov = res.model_cost.overhead_breakdown()

    return {
        # --- knobs (parallel-coordinates input axes)
        "weight_quant_bits": cfg.weight_bits,
        "activation_quant_bits": cfg.activation_bits,
        "sparsity": cfg.sparsity,
        "depthwise_bits": cfg.depthwise_bits,
        "weight_method": cfg.weight_method,
        "huffman": cfg.use_huffman,
        "qat": qat,
        # --- outcomes (parallel-coordinates output axes)
        "compression_ratio": s["model_compression_ratio"],
        "weight_compression_ratio": s["weight_compression_ratio"],
        "activation_compression_ratio": s.get("activation_compression_ratio", 1.0),
        "model_size_mb": s["compressed_mb"],
        "quantized_acc": acc["top1"],
        "baseline_acc": base_top1,
        "acc_drop": base_top1 - acc["top1"],
        "top5": acc["top5"],
        # --- storage accounting
        "overhead_share": ov["overhead_share"],
        "value_bits": ov["value_bits"],
        "index_bits": ov["index_bits"],
        "metadata_bits": ov["metadata_bits"],
        "seconds": time.time() - t0,
        "qat_history": hist,
    }


def build_grid(args) -> List[CompressionConfig]:
    """The configurations to sweep.

    The grid varies the three knobs that dominate the trade-off: weight bits,
    activation bits and sparsity. Depthwise and edge layers are held at 8 bits
    throughout, since they are the documented exceptions to the compression
    policy (2.9% and 0.6% of parameters respectively) and varying them would
    obscure the axes that matter.
    """
    if args.quick:
        wbits, abits, sparsities = (4, 8), (4, 8), (0.0, 0.8)
    else:
        wbits = tuple(int(b) for b in args.weight_bits.split(","))
        abits = tuple(int(b) for b in args.activation_bits.split(","))
        sparsities = tuple(float(s) for s in args.sparsities.split(","))

    # itertools.product is the full cross product: 5 weight widths x 4
    # activation widths x 3 sparsities = 60 configurations. A full grid rather
    # than a random search, because the parallel-coordinates chart is only
    # readable when every axis is evenly covered.
    return [
        CompressionConfig(weight_bits=w, activation_bits=a, sparsity=sp,
                          depthwise_bits=args.depthwise_bits, edge_bits=8, bn_bits=8,
                          weight_method=args.weight_method, use_huffman=not args.no_huffman)
        for w, a, sp in itertools.product(wbits, abits, sparsities)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
    ap.add_argument("--weight-bits", default="2,3,4,6,8")
    ap.add_argument("--activation-bits", default="2,4,6,8")
    ap.add_argument("--sparsities", default="0.0,0.5,0.8")
    ap.add_argument("--depthwise-bits", type=int, default=8)
    ap.add_argument("--weight-method", default="kmeans")
    ap.add_argument("--no-huffman", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--qat-lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="cs6886-a2-compression-sweep")
    ap.add_argument("--results", default="results/sweep_results.json")
    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = get_device()

    loaders = build_loaders(DataConfig(seed=args.seed), download=False)
    _, test_loader, _ = loaders

    model = mobilenet_v2_cifar().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    base = evaluate(model, test_loader, device)
    print(f"fp32 baseline: top-1 {base['top1']:.2f}%  (checkpoint epoch {ckpt.get('epoch')})")

    grid = build_grid(args)
    qcfg = QATConfig(epochs=args.qat_epochs, lr=args.qat_lr)
    print(f"sweeping {len(grid)} configurations, qat={args.qat}\n")

    hdr = (f"{'w':>3}{'a':>3}{'sp':>6}{'  ratio':>9}{'w-ratio':>9}{'a-ratio':>9}"
           f"{'MB':>8}{'top-1':>8}{'drop':>8}{'ovh%':>7}{'s':>6}")
    print(hdr); print("-" * len(hdr))

    results = []
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)

    for cfg in grid:
        r = run_one(model, cfg, loaders, device, base["top1"], qat=args.qat, qcfg=qcfg)
        results.append(r)

        print(f"{r['weight_quant_bits']:>3}{r['activation_quant_bits']:>3}"
              f"{r['sparsity']:>6.2f}{r['compression_ratio']:>9.2f}"
              f"{r['weight_compression_ratio']:>9.2f}"
              f"{r['activation_compression_ratio']:>9.2f}{r['model_size_mb']:>8.3f}"
              f"{r['quantized_acc']:>8.2f}{r['acc_drop']:>8.2f}"
              f"{100*r['overhead_share']:>7.1f}{r['seconds']:>6.0f}", flush=True)

        if args.wandb:
            import wandb
            run = wandb.init(project=args.wandb_project, reinit=True,
                             name=f"w{cfg.weight_bits}_a{cfg.activation_bits}_sp{cfg.sparsity:g}"
                                  + ("_qat" if args.qat else ""),
                             config={k: v for k, v in r.items()
                                     if k in ("weight_quant_bits", "activation_quant_bits",
                                              "sparsity", "depthwise_bits", "weight_method",
                                              "huffman", "qat")})
            run.log({k: v for k, v in r.items() if k != "qat_history"})
            run.summary.update({k: v for k, v in r.items() if k != "qat_history"})
            run.finish()

        with open(args.results, "w") as f:
            json.dump({"baseline": base, "results": results,
                       "args": vars(args)}, f, indent=2)

    # Best configuration by accuracy at each compression level, for the report.
    # A configuration is on the Pareto front when nothing else beats it on BOTH
    # axes at once. Walking the list from the highest compression ratio
    # downwards, a configuration joins the front only if it is more accurate
    # than everything already there: anything already accepted has a higher
    # ratio, so being less accurate than it means being worse on both counts.
    print("\nPareto front (no other configuration is both smaller and more accurate):")
    front = []
    for r in sorted(results, key=lambda r: -r["compression_ratio"]):
        if all(r["quantized_acc"] > f["quantized_acc"] for f in front):
            front.append(r)
    for r in front:
        print(f"  w{r['weight_quant_bits']} a{r['activation_quant_bits']} "
              f"sp{r['sparsity']:.2f}  ratio {r['compression_ratio']:6.2f}x  "
              f"{r['model_size_mb']:.3f} MB  top-1 {r['quantized_acc']:.2f}%")


if __name__ == "__main__":
    main()

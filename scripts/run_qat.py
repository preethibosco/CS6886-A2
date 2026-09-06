"""Prune-and-retrain + quantization-aware fine-tuning at the aggressive settings.

One-shot pruning above 70% destroys the network (top-1 falls to 13-66%). Deep
Compression's answer is to retrain: the surviving weights adapt to compensate
for the removed ones, and the network adapts to the quantization grid at the
same time. This script measures how much of that loss retraining recovers.

Run:  python scripts/run_qat.py --configs 0.7,0.8,0.9 --epochs 20
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.compress.pipeline import CompressionConfig, compress
from src.compress.qat import QATConfig, finetune
from src.data import DataConfig, build_loaders
from src.evaluate import evaluate
from src.models.mobilenetv2 import mobilenet_v2_cifar
from src.utils import get_device, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
ap.add_argument("--configs", default="0.7,0.8,0.9",
                help="sparsities to run at the fixed --weight-bits/--activation-bits")
ap.add_argument("--grid", default="",
                help="explicit 'wbits,abits,sparsity;...' triples, overrides --configs")
ap.add_argument("--weight-bits", type=int, default=4)
ap.add_argument("--activation-bits", type=int, default=8)
ap.add_argument("--method", default="linear_per_tensor")
ap.add_argument("--depthwise-bits", type=int, default=8,
                help="depthwise convolutions; 15.4% of the compressed model at w3/sp0.8")
ap.add_argument("--bn-bits", type=int, default=8,
                help="BatchNorm params and buffers; 18.2% of the compressed model at w3/sp0.8")
ap.add_argument("--edge-bits", type=int, default=8, help="stem convolution and classifier")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--grad-clip", type=float, default=5.0,
                help="global gradient-norm clip; 0 disables. Required at low bit "
                     "width - see scripts/diagnose_qat_collapse.py")
ap.add_argument("--max-layer-sparsity", type=float, default=0.95,
                help="cap on any single layer's sparsity under the global threshold")
ap.add_argument("--out", default="results/qat_results.json")
args = ap.parse_args()

seed_everything(42, deterministic=False)   # fine-tuning: allow cudnn autotuning
device = get_device()
loaders = build_loaders(DataConfig(seed=42), download=False)
train_loader, test_loader, calib_loader = loaders

model = mobilenet_v2_cifar().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                 weights_only=False)["model"])
base = evaluate(model, test_loader, device)
print(f"fp32 baseline top-1 = {base['top1']:.2f}%")
print(f"method={args.method}  w{args.weight_bits} a{args.activation_bits}  "
      f"QAT {args.epochs} epochs @ lr {args.lr}\n")

if args.grid:
    points = [tuple(float(v) for v in t.split(",")) for t in args.grid.split(";") if t]
    points = [(int(w), int(a), sp) for w, a, sp in points]
else:
    points = [(args.weight_bits, args.activation_bits, float(x))
              for x in args.configs.split(",")]

out = []
for wb, ab, sp in points:
    print(f"=== w{wb} a{ab} sparsity {sp:.2f} ===", flush=True)
    cfg = CompressionConfig(weight_bits=wb, activation_bits=ab,
                            depthwise_bits=args.depthwise_bits,
                            edge_bits=args.edge_bits, bn_bits=args.bn_bits, sparsity=sp,
                            max_layer_sparsity=args.max_layer_sparsity,
                            weight_method=args.method, use_huffman=True)
    qcfg = QATConfig(epochs=args.epochs, lr=args.lr, grad_clip=args.grad_clip)

    # PTQ reference: the same configuration without any fine-tuning.
    ptq_model, ptq_res = compress(model, cfg, calib_loader, device)
    ptq_acc = evaluate(ptq_model, test_loader, device)

    t0 = time.time()
    tuned, hist = finetune(model, cfg, qcfg, train_loader, test_loader,
                           calib_loader, device,
                           log_fn=lambda r: print(
                               f"    epoch {r['epoch']:2d}/{args.epochs}  "
                               f"train {r['train_top1']:.2f}  test {r['test_top1']:.2f}",
                               flush=True))
    cmodel, res = compress(tuned, cfg, calib_loader, device)
    acc = evaluate(cmodel, test_loader, device)
    s = res.summary

    rec = {"sparsity": sp, "weight_bits": wb,
           "activation_bits": ab, "method": args.method,
           "depthwise_bits": args.depthwise_bits, "bn_bits": args.bn_bits,
           "edge_bits": args.edge_bits,
           "ptq_top1": ptq_acc["top1"], "qat_top1": acc["top1"],
           "baseline_top1": base["top1"],
           "model_ratio": s["model_compression_ratio"],
           "weight_ratio": s["weight_compression_ratio"],
           "activation_ratio": s.get("activation_compression_ratio"),
           "size_mb": s["compressed_mb"], "minutes": (time.time() - t0) / 60,
           "history": hist}
    out.append(rec)
    print(f"\n  w{wb} a{ab} sp={sp:.2f}  ratio {s['model_compression_ratio']:.2f}x  "
          f"{s['compressed_mb']:.3f} MB   PTQ {ptq_acc['top1']:.2f}% -> "
          f"QAT {acc['top1']:.2f}%   (recovered {acc['top1']-ptq_acc['top1']:+.2f} pts, "
          f"{acc['top1']-base['top1']:+.2f} vs fp32)   [{(time.time()-t0)/60:.1f} min]\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"baseline": base, "results": out, "args": vars(args)},
              open(args.out, "w"), indent=2)

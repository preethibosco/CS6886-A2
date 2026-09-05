"""Rate-distortion sweep: does the entropy penalty beat both k-means and linear?

Run:  python scripts/test_ecsq.py --checkpoint <path>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.compress.huffman import huffman_result_from_counts
from src.compress.quantize import compute_qparams, dequantize, quantize
from src.compress.weight_share import share_weights
from src.models.mobilenetv2 import mobilenet_v2_cifar

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
ap.add_argument("--bits", type=int, default=4)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = mobilenet_v2_cifar().to(dev)
m.load_state_dict(torch.load(args.checkpoint, map_location=dev, weights_only=False)["model"])
b = args.bits
mods = dict(m.named_modules())

def cnt(c, a):
    h = torch.bincount(c.flatten().long(), minlength=a)
    return {i: int(v) for i, v in enumerate(h.tolist()) if v}

for lname in ("features.11.conv.2", "features.17.conv.2", "features.18.0"):
    w = mods[lname].weight.data
    n = w.numel()
    print(f"\n{lname}  ({n:,} weights, b={b}, fixed-width = {n*b:,} bits)")
    hdr = f"  {'method':<22}{'MSE':>11}{'entropy':>9}{'used codes':>12}{'total bits':>12}{'vs fixed':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    s_, z_ = compute_qparams(w, b, "symmetric", "per_tensor", channel_dim=0)
    q = quantize(w, s_, z_, b, "symmetric", channel_dim=0)
    mse = ((w - dequantize(q, s_, z_, channel_dim=0)) ** 2).mean().item()
    c = cnt(q + 2 ** (b - 1) - 1, 2 ** b)
    hr = huffman_result_from_counts(c, b, 2 ** b); tot = hr.total_bits + 32
    print(f"  {'linear/tensor':<22}{mse:>11.3e}{hr.entropy:>9.3f}{len(c):>12}{tot:>12,}{100*(1-tot/(n*b)):>9.1f}%")

    for lam in (0.0, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4):
        r = share_weights(w, b, lam=lam)
        mse = ((w - r.reconstructed) ** 2).mean().item()
        c = cnt(r.indices, 2 ** b)
        hr = huffman_result_from_counts(c, b, 2 ** b)
        tot = hr.total_bits + r.codebook_bits
        tag = "kmeans (lam=0)" if lam == 0 else f"ECSQ lam={lam:g}"
        print(f"  {tag:<22}{mse:>11.3e}{hr.entropy:>9.3f}{len(c):>12}{tot:>12,}{100*(1-tot/(n*b)):>9.1f}%")

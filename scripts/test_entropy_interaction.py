"""Does k-means weight sharing fight Huffman coding?

k-means minimises inertia by spreading weights evenly across all 2^b clusters,
which drives the code distribution toward uniform - and a uniform distribution
is exactly the one Huffman cannot compress. Linear quantization sets its grid
from the tensor extremes, so most weights fall on a few codes near zero, giving
a low-entropy stream that Huffman exploits.

If that is right, then stage 2 (weight sharing) and stage 3 (entropy coding)
interact, and optimising stage 2 for reconstruction error alone is a mistake.

Run:  python scripts/test_entropy_interaction.py --checkpoint <path>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.compress.huffman import huffman_result_from_counts
from src.compress.quantize import compute_qparams, quantize
from src.compress.weight_share import share_weights
from src.models.mobilenetv2 import mobilenet_v2_cifar

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
ap.add_argument("--bits", type=int, default=4)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2_cifar().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                 weights_only=False)["model"])
b = args.bits

layers = ["features.5.conv.0.0", "features.11.conv.2", "features.17.conv.2", "features.18.0"]
mods = dict(model.named_modules())

def counts_of(codes, alpha):
    h = torch.bincount(codes.flatten().long(), minlength=alpha)
    return {i: int(c) for i, c in enumerate(h.tolist()) if c}

hdr = (f"{'layer':<22}{'method':<16}{'MSE':>11}{'entropy':>9}{'huff b/w':>10}"
       f"{'total bits':>12}{'vs fixed':>10}")
print(f"weight bit width b = {b}  (fixed-width cost is {b}.00 bits/weight)\n")
print(hdr); print("-" * len(hdr))

for lname in layers:
    w = mods[lname].weight.data
    n = w.numel()
    rows = []

    r = share_weights(w, b)
    rows.append(("kmeans", ((w - r.reconstructed) ** 2).mean().item(),
                 counts_of(r.indices, 2 ** b), r.codebook_bits))

    for gran, tag in (("per_channel", "linear/channel"), ("per_tensor", "linear/tensor")):
        s, z = compute_qparams(w, b, "symmetric", gran, channel_dim=0)
        q = quantize(w, s, z, b, "symmetric", channel_dim=0)
        from src.compress.quantize import dequantize
        recon = dequantize(q, s, z, channel_dim=0)
        codes = q + (2 ** (b - 1) - 1)
        meta = (s.numel() if s.ndim else 1) * 32
        rows.append((tag, ((w - recon) ** 2).mean().item(), counts_of(codes, 2 ** b), meta))

    for i, (tag, mse, cnts, meta) in enumerate(rows):
        hr = huffman_result_from_counts(cnts, b, 2 ** b)
        total = hr.total_bits + meta
        print(f"{lname if i == 0 else '':<22}{tag:<16}{mse:>11.3e}{hr.entropy:>9.3f}"
              f"{hr.mean_length:>10.3f}{total:>12,}{100*(1-total/(n*b)):>9.1f}%")
    print()

print("Reading: 'entropy' is the information content of the code stream in bits/symbol.")
print("A stream at entropy b is incompressible; the gap b - entropy is what Huffman can win.")

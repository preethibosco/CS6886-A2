"""Verify the sparse encoders: exact round-trip + measured bitmap/relative crossover.

Run:  python scripts/test_prune_encoding.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.compress.prune import (choose_best_encoding, encode_bitmap,
                                encode_relative_index, magnitude_mask,
                                verify_roundtrip)

torch.manual_seed(0)

print("=== round-trip correctness (decoder must reproduce the masked tensor exactly) ===")
all_ok = True
for shape in [(64, 32, 1, 1), (320, 960, 1, 1), (10, 1280)]:
    w = torch.randn(*shape)
    for sparsity in (0.0, 0.5, 0.8, 0.95):
        mask = magnitude_mask(w, sparsity)
        codes = torch.round(w * 7)          # stand-in for quantized codes
        res = verify_roundtrip(codes, mask, value_bits=4)
        ok = all(res.values())
        all_ok &= ok
        print(f"  {str(shape):<16} sparsity={sparsity:<5} "
              f"bitmap={res['bitmap']} relative={res['relative']}  "
              f"{'OK' if ok else 'FAIL'}")
print(f"  -> {'ALL ROUND-TRIPS EXACT' if all_ok else 'FAILURE'}\n")

print("=== measured crossover: which encoding is cheaper, and where ===")
w = torch.randn(320, 960, 1, 1)          # a real MobileNet-v2 layer shape
codes = torch.round(w * 7)
n = w.numel()
print(f"  layer {tuple(w.shape)} = {n:,} weights, 4-bit value codes")
hdr = f"  {'sparsity':>9}{'nnz':>10}{'bitmap bits':>14}{'relative bits':>15}{'entries':>10}{'fillers':>9}{'winner':>11}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for sp in (0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98):
    mask = magnitude_mask(w, sp)
    bm = encode_bitmap(codes, mask, 4)
    ri = encode_relative_index(codes, mask, 4, index_bits=4)
    best = choose_best_encoding(codes, mask, 4)
    entries = len(ri.payload["deltas"])
    print(f"  {sp:>9.2f}{bm.nnz:>10,}{bm.num_bits:>14,}{ri.num_bits:>15,}"
          f"{entries:>10,}{entries - ri.nnz:>9,}{best.kind:>11}")

print("\n  dense fp32 baseline for this layer: "
      f"{n * 32:,} bits ({n * 4 / 2**20:.3f} MB)")

print("\n=== delta-width search: fillers vs entry cost, per sparsity ===")
hdr = (f"  {'sparsity':>9}{'nnz':>9} | " +
       " ".join(f"{'ib=' + str(b):>12}" for b in (3, 4, 5, 6, 8)) +
       f" | {'best':>16}{'vs ib=4':>10}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for sp in (0.7, 0.8, 0.9, 0.95, 0.98):
    mask = magnitude_mask(w, sp)
    cells, costs = [], {}
    for ib in (3, 4, 5, 6, 8):
        e = encode_relative_index(codes, mask, 4, index_bits=ib)
        fill = len(e.payload["deltas"]) - e.nnz
        costs[ib] = e.num_bits
        cells.append(f"{e.num_bits:>7,}/{fill:<4,}")
    best = choose_best_encoding(codes, mask, 4)
    gain = 100 * (1 - best.num_bits / costs[4])
    label = best.kind + (f"(ib={best.payload.get('index_bits')})" if best.kind == "relative" else "")
    print(f"  {sp:>9.2f}{int(mask.sum()):>9,} | " + " ".join(f"{c:>12}" for c in cells) +
          f" | {label:>16}{gain:>9.1f}%")
print("  (cells are: total bits / filler entries)")

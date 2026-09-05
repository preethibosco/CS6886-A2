"""Verify the Huffman coder: exact round-trip, prefix-freeness, entropy bound.

Run:  python scripts/test_huffman.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.compress.huffman import (build_huffman, decode, encode, entropy_bits,
                                  huffman_compress)

torch.manual_seed(0)

print("=== correctness ===")
cases = {
    "uniform 4-bit":        torch.randint(0, 16, (50_000,)),
    "gaussian 4-bit codes": torch.round(torch.randn(50_000) * 3).clamp(-7, 7).long() + 7,
    "gaussian 8-bit codes": torch.round(torch.randn(50_000) * 40).clamp(-127, 127).long() + 127,
    "highly skewed":        torch.cat([torch.zeros(45_000), torch.randint(1, 16, (5_000,))]).long(),
    "single symbol":        torch.zeros(1_000, dtype=torch.long),
}
for name, data in cases.items():
    syms = data.tolist()
    code = build_huffman(syms, alphabet_size=256)
    bits = encode(syms, code)
    back = decode(bits, code, len(syms))
    exact = back == syms
    # prefix-freeness: no code may be a prefix of another
    cs = sorted(code.codes.values())
    prefix_free = all(not cs[i + 1].startswith(cs[i]) for i in range(len(cs) - 1))
    print(f"  {name:<22} roundtrip={'OK ' if exact else 'FAIL'} "
          f"prefix_free={'OK ' if prefix_free else 'FAIL'} "
          f"symbols={len(code.codes):>3}")

print("\n=== entropy bound: H <= mean length < H + 1 ===")
hdr = f"  {'stream':<22}{'H (bits)':>10}{'mean len':>10}{'gap':>8}{'bound ok':>10}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, data in cases.items():
    syms = data.tolist()
    r = huffman_compress(syms, fixed_width_bits=8, alphabet_size=256)
    ok = r.entropy <= r.mean_length < r.entropy + 1 + 1e-9
    print(f"  {name:<22}{r.entropy:>10.4f}{r.mean_length:>10.4f}"
          f"{r.mean_length - r.entropy:>8.4f}{'OK' if ok else 'FAIL':>10}")

print("\n=== saving vs fixed-width, with the code table charged ===")
hdr = (f"  {'stream':<22}{'fixed bits':>12}{'payload':>11}{'table':>8}"
       f"{'total':>11}{'saving':>9}{'verified':>10}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, data in cases.items():
    syms = data.tolist()
    b = 4 if "4-bit" in name or "skewed" in name or "single" in name else 8
    r = huffman_compress(syms, fixed_width_bits=b, alphabet_size=2 ** b)
    saving = 100 * (1 - r.total_bits / r.original_bits)
    print(f"  {name:<22}{r.original_bits:>12,}{r.payload_bits:>11,}{r.table_bits:>8,}"
          f"{r.total_bits:>11,}{saving:>8.1f}%{str(r.verified):>10}")

print("\n=== when Huffman LOSES: small stream, table dominates ===")
for n in (32, 128, 512, 4096, 32768):
    syms = torch.randint(0, 16, (n,)).tolist()
    r = huffman_compress(syms, fixed_width_bits=4, alphabet_size=16)
    saving = 100 * (1 - r.total_bits / r.original_bits)
    verdict = "use huffman" if r.total_bits < r.original_bits else "store raw"
    print(f"  n={n:<7,} fixed={r.original_bits:>8,}  payload={r.payload_bits:>8,}"
          f"  table={r.table_bits:>4}  total={r.total_bits:>8,}  {saving:>6.1f}%  -> {verdict}")

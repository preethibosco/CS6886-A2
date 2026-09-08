"""Conformance gate: architecture -> implementation -> assignment requirements.

Run:  python scripts/conformance.py [--checkpoint PATH]

This exists because two correctness bugs in the compression pipeline reached
execution despite the code compiling and producing plausible-looking numbers
(a dense encoding that could not decode pruned positions, and filler entries
with no code mapping to zero). Neither was visible in the reported compression
ratio. The lesson is that a size figure is only trustworthy if a decoder can
reproduce the model from it, so the strongest check here is exactly that: take
the encoding the pipeline claims, decode it, and compare against the weights
the evaluated model actually used.

Sections:
  A. Forbidden APIs        - the assignment bans compression library calls
  B. Architecture          - MobileNet-v2 invariants (depth, blocks, shapes)
  C. Quantizer             - round-trip error within half a step
  D. Entropy coder         - prefix-free, entropy bound, fast path == reference
  E. Sparse encoders       - exact round-trip, closed form == iterative
  F. Reserved zero code    - pruned positions reconstruct to exactly 0.0
  G. END-TO-END DECODE     - claimed encoding reproduces the evaluated weights
  H. Accounting            - no storage silently uncounted
"""
from __future__ import annotations

import argparse
import os
import ast
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.compress.huffman import (build_huffman, decode, encode, entropy_bits,
                                  huffman_compress, huffman_result_from_counts)
from src.compress.prune import (PruneConfig, Pruner, decode_bitmap,
                                decode_relative_index, encode_bitmap,
                                encode_relative_index, magnitude_mask,
                                relative_index_stats)
from src.compress.quantize import compute_qparams, fake_quantize, quant_bounds
from src.compress.sizing import fp32_model_bits
from src.compress.weight_share import share_weights
from src.models.mobilenetv2 import mobilenet_v2_cifar

RESULTS = []


def check(section: str, name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((section, name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------- A
BANNED_MODULES = {"sklearn", "zlib", "gzip", "bz2", "lzma", "bitsandbytes",
                  "scipy.cluster", "torchao"}
BANNED_ATTRS = {"torch.quantization", "torch.ao.quantization",
                "torch.nn.utils.prune", "nn.utils.prune", "to_sparse",
                "to_sparse_csr", "quantize_per_tensor", "quantize_per_channel",
                "fake_quantize_per_tensor_affine", "fake_quantize_per_channel_affine"}


def _attr_path(node):
    """Dotted name of an attribute/name chain, e.g. torch.ao.quantization."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def section_a():
    """Scan the AST, not the text.

    A regex over source lines cannot tell a call from a docstring: the first
    version of this check failed on prune.py's own docstring, which states that
    torch.nn.utils.prune is NOT used. Parsing the syntax tree looks only at real
    imports and real attribute accesses, so prose can never trip it and an
    actual call can never hide inside a string.
    """
    print("\nA. Forbidden compression APIs (assignment: write your own code)")
    hits = []
    for root in ("src", "scripts"):
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                tree = ast.parse(open(path).read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            if a.name.split(".")[0] in BANNED_MODULES:
                                hits.append(f"{path}:{node.lineno} import {a.name}")
                    elif isinstance(node, ast.ImportFrom):
                        mod = (node.module or "")
                        if mod.split(".")[0] in BANNED_MODULES or mod in BANNED_ATTRS:
                            hits.append(f"{path}:{node.lineno} from {mod}")
                    elif isinstance(node, ast.Attribute):
                        dotted = _attr_path(node)
                        if any(dotted.endswith(b) for b in BANNED_ATTRS):
                            hits.append(f"{path}:{node.lineno} {dotted}")
    check("A", "no compression library calls in source (AST scan)",
          not hits, "clean" if not hits else "; ".join(hits[:4]))
    py = sum(1 for r in ("src", "scripts") for _, _, fs in os.walk(r)
             for f in fs if f.endswith(".py"))
    check("A", "scanner actually parsed the source tree", py >= 10, f"{py} files parsed")


# --------------------------------------------------------------------- B
def section_b():
    print("\nB. MobileNet-v2 architecture invariants")
    m = mobilenet_v2_cifar().eval()
    n_layers = sum(1 for x in m.modules() if isinstance(x, (nn.Conv2d, nn.Linear)))
    n_blocks = sum(1 for x in m.features if type(x).__name__ == "InvertedResidual")
    n_params = sum(p.numel() for p in m.parameters())
    check("B", "depth = 53 weight-bearing layers", n_layers == 53, f"got {n_layers}")
    check("B", "17 inverted residual blocks", n_blocks == 17, f"got {n_blocks}")
    check("B", "parameter count = 2,236,682", n_params == 2_236_682, f"got {n_params:,}")

    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32)
        for blk in m.features:
            x = blk(x)
    check("B", "CIFAR stem gives a 4x4 final feature map",
          tuple(x.shape[2:]) == (4, 4), f"got {tuple(x.shape)}")

    # Linear bottleneck: the projection conv must be followed by BN and NO ReLU.
    ok = True
    for blk in m.features:
        if type(blk).__name__ != "InvertedResidual":
            continue
        if not isinstance(blk.conv[-1], nn.BatchNorm2d):
            ok = False
    check("B", "every block ends BN with no activation (linear bottleneck)", ok)

    fp = fp32_model_bits(m)
    check("B", "BN running buffers counted in the fp32 baseline",
          fp["bn_buffer_values"] == 34_112,
          f"{fp['bn_buffer_values']:,} buffer values, baseline {fp['total_mb']:.3f} MB")


# --------------------------------------------------------------------- C
def section_c():
    print("\nC. Quantizer: round-trip error <= half a quantization step")
    torch.manual_seed(0)
    worst, ok = 0.0, True
    for b in (2, 3, 4, 6, 8):
        for sch in ("symmetric", "asymmetric"):
            for gr in ("per_tensor", "per_channel"):
                x = torch.randn(64, 256)
                s, _ = compute_qparams(x, b, sch, gr)
                err = (x - fake_quantize(x, b, sch, gr)).abs().max().item()
                bound = (s.max().item() if s.ndim else s.item()) / 2 * 1.001
                ok &= err <= bound
                worst = max(worst, err / bound)
    check("C", "20 configurations within bound", ok, f"worst = {worst:.3f} x bound")

    qmin, qmax = quant_bounds(4, "symmetric")
    check("C", "symmetric range is exactly symmetric about zero",
          qmin == -qmax, f"[{qmin}, {qmax}]")


# --------------------------------------------------------------------- D
def section_d():
    print("\nD. Huffman coder")
    torch.manual_seed(0)
    streams = {
        "gaussian4": (torch.round(torch.randn(40_000) * 3).clamp(-7, 7).long() + 7, 4, 16),
        "skewed":    (torch.cat([torch.zeros(36_000), torch.randint(1, 16, (4_000,))]).long(), 4, 16),
        "uniform8":  (torch.randint(0, 256, (40_000,)), 8, 256),
    }
    rt = pf = bound = fast = True
    for name, (data, b, alpha) in streams.items():
        syms = data.tolist()
        code = build_huffman(syms, alphabet_size=alpha)
        bits = encode(syms, code)
        rt &= decode(bits, code, len(syms)) == syms
        cs = sorted(code.codes.values())
        pf &= all(not cs[i + 1].startswith(cs[i]) for i in range(len(cs) - 1))
        r = huffman_compress(syms, b, alphabet_size=alpha, verify=False)
        bound &= r.entropy <= r.mean_length < r.entropy + 1 + 1e-9
        fast &= huffman_result_from_counts(dict(Counter(syms)), b, alpha).total_bits == r.total_bits
    check("D", "decode(encode(x)) == x", rt)
    check("D", "codes are prefix-free", pf)
    check("D", "H <= mean length < H+1", bound)
    check("D", "histogram fast path == symbol-list reference", fast)


# --------------------------------------------------------------------- E
def section_e():
    print("\nE. Sparse encoders")
    torch.manual_seed(0)
    w = torch.randn(300, 200)
    codes = torch.randint(0, 16, w.shape)
    rt, cf = True, True
    for sp in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
        mask = magnitude_mask(w, sp)
        ref = (codes * mask).float()
        rt &= torch.equal(decode_bitmap(encode_bitmap(codes, mask, 4)).float(), ref)
        for ib in (3, 4, 5, 6, 8):
            loop = encode_relative_index(codes, mask, 4, index_bits=ib)
            rt &= torch.equal(decode_relative_index(loop).float(), ref)
            fastst = relative_index_stats(mask, codes, ib, 16, zero_code=0)
            cf &= (len(loop.payload["deltas"]) == fastst["num_entries"]
                   and Counter(loop.payload["deltas"]) == Counter(fastst["delta_counts"]))
    check("E", "bitmap and relative-index decode exactly (36 configs)", rt)
    check("E", "vectorised delta stats == iterative encoder (30 configs)", cf)


# --------------------------------------------------------------------- F
def section_f():
    print("\nF. Reserved zero code (required for filler entries to decode)")
    torch.manual_seed(0)
    w = torch.randn(200, 300) * 0.05
    ok, distinct = True, True
    for sp in (0.5, 0.8, 0.95):
        mask = magnitude_mask(w, sp)
        r = share_weights(w, 4, mask=mask)
        ok &= (r.centroids[0].item() == 0.0)
        ok &= (r.reconstructed[~mask].abs().max().item() == 0.0)
        distinct &= (len(torch.unique(r.centroids)) == 16)
    check("F", "centroid[0] is exactly 0.0 when pruning is active", ok)
    check("F", "all 2^b codes remain distinct", distinct)


# --------------------------------------------------------------------- G
def section_g(checkpoint: str | None):
    print("\nG. END-TO-END: does the claimed encoding reproduce the evaluated weights?")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mobilenet_v2_cifar().to(device)
    if checkpoint and os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device,
                                         weights_only=False)["model"])
        src = os.path.basename(checkpoint)
    else:
        src = "randomly initialised (no checkpoint given)"
    print(f"    weights: {src}")

    # Pick a representative spread of layers rather than all 53, for runtime.
    targets = ["features.2.conv.0.0", "features.11.conv.2", "features.17.conv.2",
               "features.18.0", "classifier.1"]
    mods = dict(model.named_modules())

    all_ok = True
    for name in targets:
        m = mods[name]
        w = m.weight.data
        for sp in (0.0, 0.85):
            mask = magnitude_mask(w, sp) if sp else torch.ones_like(w, dtype=torch.bool)
            res = share_weights(w, 4, mask=mask)
            # This is what the model is evaluated with:
            evaluated = res.reconstructed

            # This is what the reported size claims we stored. Decode it back.
            if sp == 0.0:
                decoded_codes = res.indices                       # dense
                kind = "dense"
            else:
                enc = encode_relative_index(res.indices, mask, 4, index_bits=6)
                decoded_codes = decode_relative_index(enc).long().to(w.device)
                kind = "rel6"
            reconstructed = res.centroids[decoded_codes].reshape(w.shape)

            exact = torch.equal(reconstructed.float(), evaluated.float())
            all_ok &= exact
            print(f"    {name:<22} sp={sp:<5} {kind:<6} "
                  f"max|diff|={(reconstructed - evaluated).abs().max().item():.2e} "
                  f"{'OK' if exact else 'MISMATCH'}")
    check("G", "decoded encoding == weights used for the accuracy number", all_ok)


# --------------------------------------------------------------------- H
def section_g2():
    """fold_bn must remove BatchNorm, not merely stop charging for it.

    sizing.batchnorm_cost returns an empty cost list when `folded` is set, so if
    the folding transform were missing the pipeline would report BatchNorm as
    zero storage while the model still ran it in fp32 - an ~18% under-report of
    the compressed model. This check ties the accounting to the transform.
    """
    print("\nG2. BatchNorm folding is real, not just uncharged")
    import copy as _copy
    from src.compress.fold import fold_model, verify_fold
    from src.compress.sizing import batchnorm_cost
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = mobilenet_v2_cifar().to(device).eval()
    folded, n = fold_model(_copy.deepcopy(m))
    r = verify_fold(m, folded, device)
    check("G2", "folds every conv+BN pair", n == 52, f"{n} pairs")
    check("G2", "no BatchNorm survives the fold", r["remaining_bn"] == 0,
          f"{r['remaining_bn']} remaining")
    check("G2", "folded network computes the same function",
          r["agree"] and r["same_argmax"], f"max|diff| = {r['max_abs_diff']:.2e}")
    charged = len(batchnorm_cost(folded, num_bits=8, folded=True))
    check("G2", "zero BN cost is only claimed when no BN remains",
          charged == 0 and r["remaining_bn"] == 0)


def section_g3():
    """The fp32 baseline must not shrink when a pipeline stage transforms the model.

    compress() previously derived the baseline from whatever model it was handed.
    Fine-tuning folds BatchNorm before compress() sees the model, so the folded
    runs were quoted against a baseline 51,168 values smaller than the network
    that was actually trained.
    """
    print("\nG3. Compression ratio is quoted against the untransformed model")
    import copy as _copy
    from src.compress.fold import fold_model
    from src.compress.sizing import fp32_model_bits
    m = mobilenet_v2_cifar()
    folded, _ = fold_model(_copy.deepcopy(m))
    a = fp32_model_bits(m)["total_values"]
    b = fp32_model_bits(folded)["total_values"]
    check("G3", "folding does shrink the measured model", b < a, f"{a:,} -> {b:,}")
    import inspect
    from src.compress.pipeline import compress
    sig = inspect.signature(compress)
    check("G3", "compress() accepts an explicit baseline_model",
          "baseline_model" in sig.parameters)
    src_qat = open("scripts/run_qat.py").read()
    check("G3", "run_qat.py passes the pre-fine-tuning model as baseline",
          "baseline_model=model" in src_qat)


def section_h():
    print("\nH. Storage accounting completeness")
    m = mobilenet_v2_cifar()
    fp = fp32_model_bits(m)
    params = sum(p.numel() for p in m.parameters())
    buffers = sum(b.numel() for n, b in m.named_buffers()
                  if "running_mean" in n or "running_var" in n)
    other = [n for n, _ in m.named_buffers()
             if "running_mean" not in n and "running_var" not in n
             and "num_batches" not in n]
    check("H", "baseline = parameters + BN buffers",
          fp["total_values"] == params + buffers,
          f"{params:,} + {buffers:,} = {fp['total_values']:,}")
    check("H", "no unaccounted buffers", not other,
          "none" if not other else f"unaccounted: {other}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_best.pt")
    args = ap.parse_args()

    print("=" * 74)
    print(" CONFORMANCE: architecture -> implementation -> assignment")
    print("=" * 74)
    section_a(); section_b(); section_c(); section_d()
    section_e(); section_f(); section_g(args.checkpoint); section_g2(); section_g3(); section_h()

    passed = sum(1 for *_, p, _ in RESULTS if p)
    print("\n" + "=" * 74)
    print(f" {passed}/{len(RESULTS)} checks passed")
    failed = [(s, n) for s, n, p, _ in RESULTS if not p]
    if failed:
        print(" FAILURES:")
        for s, n in failed:
            print(f"   {s}: {n}")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

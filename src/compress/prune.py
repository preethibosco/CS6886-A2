"""Magnitude-based unstructured pruning + sparse index encoding.

Stage 1 of the Deep Compression pipeline (Han et al., ICLR 2016). Written from
scratch: no torch.nn.utils.prune, no sparse-tensor library.

Two responsibilities:

  1. Deciding *which* weights to remove (`magnitude_mask`, `global_masks`,
     `Pruner`) - fine-grained (element-wise) magnitude pruning, with either a
     per-layer or a single global threshold.

  2. Deciding *how to store* the survivors (`encode_bitmap`,
     `encode_relative_index`) - because a sparse tensor is only smaller if the
     positions of the non-zeros can be encoded cheaply. Both encoders come with
     a matching decoder, and `verify_roundtrip` asserts exact reconstruction, so
     every storage number we report corresponds to a real encoding rather than
     an analytic estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Choosing what to prune
# --------------------------------------------------------------------------- #
def magnitude_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask that zeroes the `sparsity` fraction of smallest |w|.

    Fine-grained (element-wise) pruning: any individual weight may be removed,
    with no structural constraint. This gives the highest compression per unit
    of accuracy lost, at the cost of needing an index encoding to exploit -
    which is exactly the trade the encoders below quantify.

    The threshold is the `sparsity` quantile of |w|, computed via kthvalue so it
    is exact rather than interpolated.
    """
    if sparsity <= 0.0:
        return torch.ones_like(weight, dtype=torch.bool)
    if sparsity >= 1.0:
        return torch.zeros_like(weight, dtype=torch.bool)

    flat = weight.detach().abs().flatten()
    k = int(round(sparsity * flat.numel()))
    if k <= 0:
        return torch.ones_like(weight, dtype=torch.bool)
    # kthvalue gives the k-th smallest; everything strictly greater is kept.
    threshold = flat.kthvalue(k).values
    return weight.detach().abs() > threshold


def global_masks(weights: Dict[str, torch.Tensor], sparsity: float) -> Dict[str, torch.Tensor]:
    """Masks from a single threshold shared across all supplied tensors.

    Global pruning lets the *network* decide where sparsity belongs rather than
    forcing every layer to the same ratio. In MobileNet-v2 this matters a great
    deal: the late 960->160 pointwise convolutions are heavily over-parameterised
    and tolerate extreme sparsity, whereas the 3x3 depthwise filters hold only 9
    weights per channel and are damaged by even mild pruning. A global threshold
    discovers that automatically.

    Care is still needed: an unconstrained global threshold can remove a layer
    entirely. `Pruner` therefore supports a per-layer sparsity cap.
    """
    if sparsity <= 0.0:
        return {n: torch.ones_like(w, dtype=torch.bool) for n, w in weights.items()}

    all_abs = torch.cat([w.detach().abs().flatten() for w in weights.values()])
    k = int(round(sparsity * all_abs.numel()))
    if k <= 0:
        return {n: torch.ones_like(w, dtype=torch.bool) for n, w in weights.items()}
    threshold = all_abs.kthvalue(k).values
    return {n: (w.detach().abs() > threshold) for n, w in weights.items()}


@dataclass
class PruneConfig:
    """Configuration for a pruning run."""

    sparsity: float = 0.0
    # 'global' shares one threshold across layers; 'layer' applies `sparsity`
    # independently to each layer.
    scope: str = "global"
    # Layers whose parameters are never pruned. Depthwise convolutions have only
    # 9 weights per channel, so removing any of them deletes a meaningful part of
    # a channel's spatial filter; and they are only 2.9% of the model, so
    # exempting them costs almost nothing in compression ratio while protecting
    # accuracy. The stem sees the raw image and the classifier produces the
    # logits - both are tiny and both are disproportionately sensitive.
    protect_depthwise: bool = True
    protect_first_last: bool = True
    # Upper bound on any single layer's sparsity under global scope, so that the
    # global threshold can never erase a layer completely.
    max_layer_sparsity: float = 0.95


class Pruner:
    """Holds pruning masks and keeps them applied across training steps.

    Usage:
        pruner = Pruner(model, PruneConfig(sparsity=0.8))
        pruner.compute_masks()
        pruner.apply()                 # zero the pruned weights
        ... training step ...
        pruner.apply()                 # re-zero after the optimiser update

    The `apply()`-after-`step()` pattern is what makes pruning "stick" during
    fine-tuning: SGD with momentum and weight decay will otherwise drift pruned
    weights away from zero. Masking the *weights* after the step (rather than
    only the gradients) is the robust version, because momentum buffers and
    weight decay both inject updates that a gradient-only mask would miss.
    """

    def __init__(self, model: nn.Module, cfg: PruneConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.masks: Dict[str, torch.Tensor] = {}
        self.prunable = self._collect_prunable()

    def _collect_prunable(self) -> Dict[str, nn.Module]:
        """Select the modules eligible for pruning, honouring the protections."""
        convs = [(n, m) for n, m in self.model.named_modules()
                 if isinstance(m, (nn.Conv2d, nn.Linear))]
        first_name = convs[0][0]
        last_name = convs[-1][0]

        out: Dict[str, nn.Module] = {}
        for name, m in convs:
            if self.cfg.protect_depthwise and isinstance(m, nn.Conv2d) and m.groups > 1:
                continue
            if self.cfg.protect_first_last and name in (first_name, last_name):
                continue
            out[name] = m
        return out

    @torch.no_grad()
    def compute_masks(self) -> None:
        """Populate `self.masks` according to the configured scope."""
        weights = {n: m.weight for n, m in self.prunable.items()}
        if self.cfg.scope == "global":
            masks = global_masks(weights, self.cfg.sparsity)
            # Enforce the per-layer cap: if the global threshold pruned a layer
            # past the cap, restore its largest-magnitude weights until it complies.
            for name, mask in masks.items():
                w = weights[name]
                achieved = 1.0 - mask.float().mean().item()
                if achieved > self.cfg.max_layer_sparsity:
                    masks[name] = magnitude_mask(w, self.cfg.max_layer_sparsity)
            self.masks = masks
        else:
            self.masks = {n: magnitude_mask(w, self.cfg.sparsity) for n, w in weights.items()}

    @torch.no_grad()
    def apply(self) -> None:
        """Zero every pruned weight in place."""
        for name, module in self.prunable.items():
            if name in self.masks:
                module.weight.mul_(self.masks[name].to(module.weight.dtype))

    def sparsity_report(self) -> Dict[str, float]:
        """Achieved sparsity per pruned layer, plus the model-wide figure."""
        report, total, zeros = {}, 0, 0
        for name, module in self.prunable.items():
            mask = self.masks.get(name)
            if mask is None:
                continue
            n = mask.numel()
            z = n - int(mask.sum().item())
            report[name] = z / n
            total += n
            zeros += z
        report["__overall_prunable__"] = zeros / total if total else 0.0
        # Sparsity over *all* weights, including the protected ones.
        all_n = sum(m.weight.numel() for _, m in self.model.named_modules()
                    if isinstance(m, (nn.Conv2d, nn.Linear)))
        report["__overall_model__"] = zeros / all_n if all_n else 0.0
        return report


# --------------------------------------------------------------------------- #
# Storing what survives: sparse index encoding
# --------------------------------------------------------------------------- #
@dataclass
class SparseEncoding:
    """A concrete sparse encoding of one tensor, with its exact bit cost."""

    kind: str                    # 'bitmap' | 'relative' | 'dense'
    num_bits: int                # total bits for values + index metadata
    value_bits: int              # bits spent on the quantized values
    index_bits: int              # bits spent on position metadata
    nnz: int                     # number of stored (non-filler) values
    payload: dict = field(default_factory=dict)   # what a decoder needs


def encode_bitmap(codes: torch.Tensor, mask: torch.Tensor, value_bits: int) -> SparseEncoding:
    """Bitmap encoding: one presence bit per weight + value codes for survivors.

    Cost = N (bitmap) + nnz * b (values).

    Cheap metadata per *stored* value at low sparsity, but the bitmap costs one
    bit for every weight whether or not it survives, so its overhead does not
    shrink as sparsity rises.
    """
    n = mask.numel()
    nnz = int(mask.sum().item())
    values = codes.flatten()[mask.flatten()]
    return SparseEncoding(
        kind="bitmap",
        num_bits=n + nnz * value_bits,
        value_bits=nnz * value_bits,
        index_bits=n,
        nnz=nnz,
        payload={"mask": mask.detach().cpu().clone(),
                 "values": values.detach().cpu().clone(),
                 "shape": tuple(mask.shape)},
    )


def encode_relative_index(codes: torch.Tensor, mask: torch.Tensor, value_bits: int,
                          index_bits: int = 4) -> SparseEncoding:
    """Relative (delta) index encoding, as used in Deep Compression.

    Walk the flattened tensor and store, for every surviving weight, the gap to
    the previous surviving weight in `index_bits` bits alongside its value code.
    When a gap exceeds the largest representable delta (2^index_bits - 1) a
    "filler" entry is emitted: a zero value at the maximum delta, which advances
    the cursor without encoding a real weight.

    Cost = entries * (index_bits + value_bits), where entries = nnz + fillers.

    Metadata scales with the number of *survivors*, not with the tensor size, so
    this overtakes the bitmap once sparsity is high. The filler mechanism is the
    catch: at very high sparsity the gaps grow and fillers multiply, which is
    why `choose_best_encoding` compares both empirically rather than assuming.
    """
    max_delta = (1 << index_bits) - 1
    flat_mask = mask.flatten()
    flat_codes = codes.flatten()
    positions = torch.nonzero(flat_mask, as_tuple=False).flatten().tolist()

    deltas: List[int] = []
    values: List[float] = []
    prev = -1
    for pos in positions:
        gap = pos - prev
        # Emit filler entries until the remaining gap fits in index_bits.
        while gap > max_delta:
            deltas.append(max_delta)
            values.append(0.0)
            prev += max_delta
            gap = pos - prev
        deltas.append(gap)
        values.append(float(flat_codes[pos].item()))
        prev = pos

    entries = len(deltas)
    return SparseEncoding(
        kind="relative",
        num_bits=entries * (index_bits + value_bits),
        value_bits=entries * value_bits,
        index_bits=entries * index_bits,
        nnz=int(flat_mask.sum().item()),
        payload={"deltas": deltas, "values": values,
                 "shape": tuple(mask.shape), "index_bits": index_bits,
                 "numel": int(flat_mask.numel())},
    )


def decode_bitmap(enc: SparseEncoding) -> torch.Tensor:
    """Reconstruct the dense code tensor from a bitmap encoding."""
    mask = enc.payload["mask"]
    out = torch.zeros(mask.numel(), dtype=enc.payload["values"].dtype)
    out[mask.flatten()] = enc.payload["values"]
    return out.reshape(enc.payload["shape"])


def decode_relative_index(enc: SparseEncoding) -> torch.Tensor:
    """Reconstruct the dense code tensor from a relative-index encoding."""
    out = torch.zeros(enc.payload["numel"])
    prev = -1
    for delta, value in zip(enc.payload["deltas"], enc.payload["values"]):
        pos = prev + delta
        out[pos] = value
        prev = pos
    return out.reshape(enc.payload["shape"])


def choose_best_encoding(codes: torch.Tensor, mask: torch.Tensor, value_bits: int,
                         index_bits_choices: Tuple[int, ...] = (3, 4, 5, 6, 8),
                         ) -> SparseEncoding:
    """Return the cheapest encoding of this tensor, searching the delta width.

    Candidates considered:
      * dense    - no index metadata at all; wins when sparsity is low enough
                   that any positional overhead is wasted.
      * bitmap   - one presence bit per weight; overhead is fixed at N bits and
                   therefore does not shrink as sparsity rises.
      * relative - delta indices, tried at several widths.

    Searching the delta width matters. A 4-bit delta spans at most 15 positions,
    so once the mean gap between survivors exceeds that, the encoder must emit
    filler entries - and at 98% sparsity the fillers outnumber the real values
    roughly 3:1, which destroys the very compression the sparsity was bought
    for. Widening the delta to 6 or 8 bits costs a little per entry but removes
    almost all fillers. The right width depends on the achieved sparsity of the
    individual layer, so it is selected per layer by direct measurement rather
    than fixed a priori.

    The winning `kind` and width are recorded on the returned object and are
    reported per layer, since this choice is part of the storage accounting.
    """
    candidates = [
        SparseEncoding(kind="dense", num_bits=mask.numel() * value_bits,
                       value_bits=mask.numel() * value_bits, index_bits=0,
                       nnz=int(mask.sum().item()),
                       payload={"values": codes.detach().cpu().flatten().clone(),
                                "shape": tuple(mask.shape)}),
        encode_bitmap(codes, mask, value_bits),
    ]
    for ib in index_bits_choices:
        candidates.append(encode_relative_index(codes, mask, value_bits, index_bits=ib))
    return min(candidates, key=lambda e: e.num_bits)


def verify_roundtrip(codes: torch.Tensor, mask: torch.Tensor, value_bits: int,
                     index_bits: int = 4) -> Dict[str, bool]:
    """Assert that both encoders reconstruct the masked tensor exactly.

    Every storage figure in the report comes from these encoders, so this check
    is what makes those figures claims about a real encoding rather than an
    analytic guess.
    """
    reference = (codes * mask).float().cpu()
    results = {}
    bm = decode_bitmap(encode_bitmap(codes, mask, value_bits)).float()
    results["bitmap"] = torch.equal(bm, reference)
    ri = decode_relative_index(
        encode_relative_index(codes, mask, value_bits, index_bits)).float()
    results["relative"] = torch.equal(ri, reference)
    return results


def relative_index_stats(mask: torch.Tensor, codes: torch.Tensor, index_bits: int,
                         alphabet_size: int, zero_code: int = 0
                         ) -> Dict[str, object]:
    """Vectorised cost model for relative-index encoding.

    Computes, without any Python-level loop, the symbol histograms that the
    entropy coder needs:

        gap_i        = pos_i - pos_{i-1}          (pos_{-1} = -1)
        fillers_i    = (gap_i - 1) // max_delta   (closed form for the loop's
                                                   repeated subtraction)
        delta_i      = gap_i - fillers_i * max_delta   in [1, max_delta]

    The equivalence of the closed form to the iterative encoder is asserted in
    scripts/test_prune_encoding.py against `encode_relative_index`.

    Filler entries carry a zero *weight*, so their value symbol must be a code
    that decodes to exactly 0.0 - `zero_code`. This is why k-means reserves a
    codebook slot for zero whenever a pruning mask is present: without it, no
    code maps to zero and the fillers would reconstruct as the most-negative
    centroid.

    Returns a dict with the entry count and the delta / value histograms.
    """
    max_delta = (1 << index_bits) - 1
    flat_mask = mask.flatten()
    pos = torch.nonzero(flat_mask, as_tuple=False).flatten()

    if pos.numel() == 0:
        return {"num_entries": 0, "num_fillers": 0, "nnz": 0,
                "delta_counts": {}, "value_counts": {}}

    prev = torch.cat([torch.full((1,), -1, dtype=pos.dtype, device=pos.device), pos[:-1]])
    gaps = pos - prev
    fillers = (gaps - 1) // max_delta
    deltas = gaps - fillers * max_delta
    total_fillers = int(fillers.sum().item())

    delta_hist = torch.bincount(deltas, minlength=max_delta + 1)
    delta_counts = {i: int(c) for i, c in enumerate(delta_hist.tolist()) if c}
    delta_counts[max_delta] = delta_counts.get(max_delta, 0) + total_fillers

    vals = codes.flatten()[pos].long()
    value_hist = torch.bincount(vals, minlength=alphabet_size)
    value_counts = {i: int(c) for i, c in enumerate(value_hist.tolist()) if c}
    if total_fillers:
        value_counts[zero_code] = value_counts.get(zero_code, 0) + total_fillers

    return {"num_entries": int(pos.numel()) + total_fillers,
            "num_fillers": total_fillers, "nnz": int(pos.numel()),
            "delta_counts": delta_counts, "value_counts": value_counts}

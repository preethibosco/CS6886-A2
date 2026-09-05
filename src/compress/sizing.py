"""Exact storage accounting for the compressed model (Assignment Q2c, Q4).

Every number reported in the write-up is produced here. The guiding rule is
that nothing is free: if a decoder needs a value in order to reconstruct the
model, that value is charged. Concretely, the accounting includes

  * quantized weight codes (after Huffman, where Huffman actually pays),
  * sparse position metadata (bitmap or relative-index deltas),
  * the k-means codebook, or the per-channel scales/zero-points,
  * Huffman code-length tables,
  * BatchNorm gamma/beta and the running_mean/running_var buffers - which are
    NOT in model.parameters() but are required at inference,
  * biases, wherever a layer has them.

Sizes are reported in bits internally and converted to MB (2^20 bytes) at the
boundary, so rounding never accumulates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

BITS_PER_MB = 8 * 2 ** 20


@dataclass
class LayerCost:
    """Storage cost of one layer, itemised."""

    name: str
    kind: str                      # 'pointwise' | 'depthwise' | 'stem' | 'classifier' | 'bn'
    numel: int                     # weights in the dense tensor
    nnz: int                       # weights actually stored
    num_bits: int                  # bit width of the value codes
    value_bits: int = 0            # bits for the values (post-Huffman if used)
    index_bits: int = 0            # bits for sparse position metadata
    metadata_bits: int = 0         # codebook / scales / zero-points / Huffman tables
    encoding: str = "dense"        # which sparse encoding won
    huffman: bool = False          # whether entropy coding was applied
    notes: str = ""

    @property
    def total_bits(self) -> int:
        return self.value_bits + self.index_bits + self.metadata_bits

    @property
    def fp32_bits(self) -> int:
        return self.numel * 32

    @property
    def ratio(self) -> float:
        return self.fp32_bits / self.total_bits if self.total_bits else float("inf")

    @property
    def sparsity(self) -> float:
        return 1.0 - (self.nnz / self.numel) if self.numel else 0.0

    @property
    def bits_per_weight(self) -> float:
        """Effective bits per *original* weight, the honest per-layer headline."""
        return self.total_bits / self.numel if self.numel else 0.0


@dataclass
class ModelCost:
    """Total storage cost of a compressed model."""

    layers: List[LayerCost] = field(default_factory=list)

    # ---------------------------------------------------------------- totals
    @property
    def total_bits(self) -> int:
        return sum(l.total_bits for l in self.layers)

    @property
    def fp32_bits(self) -> int:
        return sum(l.fp32_bits for l in self.layers)

    @property
    def total_mb(self) -> float:
        return self.total_bits / BITS_PER_MB

    @property
    def fp32_mb(self) -> float:
        return self.fp32_bits / BITS_PER_MB

    @property
    def compression_ratio(self) -> float:
        return self.fp32_bits / self.total_bits if self.total_bits else float("inf")

    # ------------------------------------------------------------ breakdowns
    def by_kind(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for l in self.layers:
            d = out.setdefault(l.kind, {"fp32_bits": 0, "total_bits": 0, "numel": 0})
            d["fp32_bits"] += l.fp32_bits
            d["total_bits"] += l.total_bits
            d["numel"] += l.numel
        for d in out.values():
            d["ratio"] = d["fp32_bits"] / d["total_bits"] if d["total_bits"] else float("inf")
            d["share_of_compressed"] = d["total_bits"] / self.total_bits if self.total_bits else 0.0
        return out

    def overhead_breakdown(self) -> Dict[str, float]:
        """Where the compressed bits actually go.

        This is the table Q2(c) asks for: it separates the payload (value codes)
        from the metadata that makes the payload decodable.
        """
        values = sum(l.value_bits for l in self.layers)
        index = sum(l.index_bits for l in self.layers)
        meta = sum(l.metadata_bits for l in self.layers)
        total = values + index + meta
        return {
            "value_bits": values,
            "index_bits": index,
            "metadata_bits": meta,
            "total_bits": total,
            "value_share": values / total if total else 0.0,
            "index_share": index / total if total else 0.0,
            "metadata_share": meta / total if total else 0.0,
            "overhead_share": (index + meta) / total if total else 0.0,
        }

    def weights_only(self) -> "ModelCost":
        """Sub-cost covering conv/linear weights only (excludes BatchNorm).

        Q4(a) asks for the compression ratio *of the weights*, which is this
        number. The whole-model ratio is lower, because BatchNorm barely
        compresses; reporting the weight ratio as if it were the model ratio is
        the most common way these figures get overstated.
        """
        return ModelCost(layers=[l for l in self.layers if l.kind != "bn"])

    def table(self) -> str:
        hdr = (f"{'layer':<24}{'kind':<12}{'numel':>9}{'nnz':>9}{'sp%':>6}{'b':>3}"
               f"{'value':>10}{'index':>9}{'meta':>8}{'total':>10}{'b/w':>7}{'enc':<10}{'huff':>5}")
        lines = [hdr, "-" * len(hdr)]
        for l in self.layers:
            lines.append(
                f"{l.name:<24}{l.kind:<12}{l.numel:>9,}{l.nnz:>9,}{100*l.sparsity:>6.1f}"
                f"{l.num_bits:>3}{l.value_bits:>10,}{l.index_bits:>9,}{l.metadata_bits:>8,}"
                f"{l.total_bits:>10,}{l.bits_per_weight:>7.2f}{l.encoding:<10}"
                f"{'Y' if l.huffman else '-':>5}")
        lines.append("-" * len(hdr))
        lines.append(f"{'TOTAL':<24}{'':<12}{sum(l.numel for l in self.layers):>9,}"
                     f"{sum(l.nnz for l in self.layers):>9,}{'':>6}{'':>3}"
                     f"{sum(l.value_bits for l in self.layers):>10,}"
                     f"{sum(l.index_bits for l in self.layers):>9,}"
                     f"{sum(l.metadata_bits for l in self.layers):>8,}"
                     f"{self.total_bits:>10,}"
                     f"{self.total_bits/max(sum(l.numel for l in self.layers),1):>7.2f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def fp32_model_bits(model: nn.Module, include_bn_buffers: bool = True) -> Dict[str, int]:
    """Uncompressed fp32 storage of a model, itemised.

    `include_bn_buffers` controls whether running_mean / running_var are
    counted. They must be: they are not returned by `model.parameters()`, but
    inference is impossible without them. Reporting the parameter count alone
    understates the baseline by 34,112 values for MobileNet-v2 - and, worse,
    understating the *baseline* would inflate every compression ratio computed
    against it.
    """
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for n, b in model.named_buffers()
                  if "running_mean" in n or "running_var" in n)
    total = params + (buffers if include_bn_buffers else 0)
    return {
        "param_values": params,
        "bn_buffer_values": buffers,
        "total_values": total,
        "param_bits": params * 32,
        "bn_buffer_bits": buffers * 32,
        "total_bits": total * 32,
        "total_mb": total * 32 / BITS_PER_MB,
    }


def batchnorm_cost(model: nn.Module, num_bits: int = 8, folded: bool = False) -> List[LayerCost]:
    """Storage charged for BatchNorm.

    If BN has been folded into the preceding convolution the cost is zero: the
    scale and shift have been absorbed into the conv weights and bias, and
    nothing needs to be stored.

    Otherwise each BN layer stores four vectors of length C - gamma, beta,
    running_mean, running_var - quantized to `num_bits`, plus one fp32 scale and
    zero-point per vector as metadata.

    Charging BN honestly matters more than it looks. At fp32 the 68,224 BN
    values are 0.26 MB; against a compressed model of roughly 0.4 MB that is a
    dominant term, not a rounding error.
    """
    if folded:
        return []
    costs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            c = m.num_features
            values = 4 * c                    # gamma, beta, running_mean, running_var
            # 4 vectors x (fp32 scale + fp32 zero-point) of per-tensor metadata.
            meta = 4 * 2 * 32
            costs.append(LayerCost(name=name, kind="bn", numel=values, nnz=values,
                                   num_bits=num_bits, value_bits=values * num_bits,
                                   index_bits=0, metadata_bits=meta, encoding="dense",
                                   notes="gamma,beta,running_mean,running_var"))
    return costs


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #
@dataclass
class ActivationCost:
    """Activation storage, measured per single inference (batch size 1).

    Two distinct quantities are reported, because they answer different
    questions and the assignment asks us to state which we measured:

      * `total_*`  - the sum over every quantized activation tensor produced in
        one forward pass. This is the activation *traffic*: what a layer-by-layer
        accelerator moves to and from memory across an inference. It is the
        figure we quote as the activation compression ratio.

      * `peak_*`   - the largest single activation tensor. This is what sets the
        minimum on-chip buffer, and is the binding constraint on a memory-limited
        device.

    Both are computed from the same per-site bit widths, so the ratios agree;
    only the aggregation differs.
    """

    sites: List[Dict] = field(default_factory=list)

    @property
    def total_elements(self) -> int:
        return sum(s["numel"] for s in self.sites)

    @property
    def total_fp32_bits(self) -> int:
        return self.total_elements * 32

    @property
    def total_quant_bits(self) -> int:
        return sum(s["numel"] * s["bits"] for s in self.sites)

    @property
    def total_ratio(self) -> float:
        return self.total_fp32_bits / self.total_quant_bits if self.total_quant_bits else float("inf")

    @property
    def peak_elements(self) -> int:
        return max((s["numel"] for s in self.sites), default=0)

    @property
    def peak_fp32_bits(self) -> int:
        return self.peak_elements * 32

    @property
    def peak_quant_bits(self) -> int:
        return max((s["numel"] * s["bits"] for s in self.sites), default=0)

    @property
    def peak_ratio(self) -> float:
        return self.peak_fp32_bits / self.peak_quant_bits if self.peak_quant_bits else float("inf")

    def total_mb(self, quantized: bool = True) -> float:
        return (self.total_quant_bits if quantized else self.total_fp32_bits) / BITS_PER_MB

    def peak_mb(self, quantized: bool = True) -> float:
        return (self.peak_quant_bits if quantized else self.peak_fp32_bits) / BITS_PER_MB

    def table(self) -> str:
        hdr = (f"{'site':<34}{'shape':<18}{'elements':>10}{'bits':>6}"
               f"{'fp32 KB':>10}{'quant KB':>10}{'ratio':>7}")
        lines = [hdr, "-" * len(hdr)]
        for s in self.sites:
            lines.append(
                f"{s['name']:<34}{str(s.get('shape','')):<18}{s['numel']:>10,}{s['bits']:>6}"
                f"{s['numel']*32/8/1024:>10.1f}{s['numel']*s['bits']/8/1024:>10.1f}"
                f"{32/s['bits']:>7.2f}")
        lines.append("-" * len(hdr))
        lines.append(f"{'TOTAL (traffic, batch=1)':<34}{'':<18}{self.total_elements:>10,}{'':>6}"
                     f"{self.total_fp32_bits/8/1024:>10.1f}"
                     f"{self.total_quant_bits/8/1024:>10.1f}{self.total_ratio:>7.2f}")
        lines.append(f"{'PEAK (single tensor)':<34}{'':<18}{self.peak_elements:>10,}{'':>6}"
                     f"{self.peak_fp32_bits/8/1024:>10.1f}"
                     f"{self.peak_quant_bits/8/1024:>10.1f}{self.peak_ratio:>7.2f}")
        return "\n".join(lines)


def summarise(model_cost: ModelCost, act_cost: Optional[ActivationCost],
              fp32_baseline: Dict[str, int]) -> Dict[str, float]:
    """Assemble the headline numbers the assignment asks for."""
    w = model_cost.weights_only()
    ov = model_cost.overhead_breakdown()
    out = {
        "baseline_fp32_mb": fp32_baseline["total_mb"],
        "compressed_mb": model_cost.total_mb,
        "model_compression_ratio": fp32_baseline["total_bits"] / model_cost.total_bits,
        "weight_compression_ratio": w.fp32_bits / w.total_bits,
        "weights_compressed_mb": w.total_mb,
        "overhead_share": ov["overhead_share"],
        "index_bits": ov["index_bits"],
        "metadata_bits": ov["metadata_bits"],
        "value_bits": ov["value_bits"],
    }
    if act_cost is not None:
        out.update({
            "activation_compression_ratio": act_cost.total_ratio,
            "activation_peak_ratio": act_cost.peak_ratio,
            "activation_total_fp32_mb": act_cost.total_mb(quantized=False),
            "activation_total_quant_mb": act_cost.total_mb(quantized=True),
            "activation_peak_fp32_mb": act_cost.peak_mb(quantized=False),
            "activation_peak_quant_mb": act_cost.peak_mb(quantized=True),
        })
    return out

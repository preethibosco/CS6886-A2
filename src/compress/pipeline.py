"""End-to-end compression pipeline applied to MobileNet-v2 (Assignment Q2).

Composes the four hand-written stages:

    prune.py        stage 1 - fine-grained magnitude pruning + sparse encoding
    weight_share.py stage 2 - k-means weight sharing (or linear quantization)
    huffman.py      stage 3 - canonical Huffman entropy coding
    quantize.py               activation quantization (calibrated, per site)

and accounts for the result with sizing.py.

Layer policy (Q2b - which layers are compressed, and the exceptions)
-------------------------------------------------------------------
  pointwise 1x1 convolutions   pruned + quantized to `weight_bits`
      2,124,672 weights = 95.0% of the model. This is where compression has to
      happen; everything else is a rounding error by comparison.

  depthwise 3x3 convolutions   quantized to `depthwise_bits`, NOT pruned
      64,224 weights = 2.9%. Exempted from pruning and given a higher bit width
      because each channel holds only 9 weights and does no cross-channel
      mixing, so there is no redundancy to exploit and no neighbouring channel
      to absorb the error. Protecting them costs almost nothing in ratio.

  stem convolution             quantized to `edge_bits`, NOT pruned
      864 weights = 0.04%. It sees the raw image; every downstream feature
      depends on it, and it is far too small for its compression to matter.

  classifier                   quantized to `edge_bits`, NOT pruned
      12,810 weights = 0.6%. Directly produces the logits, so its error is not
      attenuated by any subsequent layer.

  BatchNorm                    quantized to `bn_bits`, or folded away
      68,224 values including the running_mean/running_var buffers. Small in
      fp32 terms but a large share of a heavily compressed model, so it is
      charged explicitly rather than ignored.

Activation policy
-----------------
  post-ReLU6 tensors     asymmetric, one-sided range [0, 6]
  post-projection / post-residual-add tensors   symmetric, signed and ~zero-mean

Ranges are calibrated on a held-out slice of the *training* set, never on the
test set.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..models.mobilenetv2 import ConvBNReLU, InvertedResidual
from .huffman import huffman_result_from_counts
from .prune import (PruneConfig, Pruner, magnitude_mask, relative_index_stats)
from .quantize import (ActivationQuantizer, ActQuantConfig, compute_qparams,
                       dequantize, quantize)
from .sizing import (ActivationCost, LayerCost, ModelCost, batchnorm_cost,
                     fp32_model_bits, summarise)
from .weight_share import share_weights


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class CompressionConfig:
    """Every knob of the compression pipeline (Q2a: "configurable").

    A single instance of this fully determines a point in the sweep, so a run
    is reproducible from its config alone.
    """

    # ---- weights
    weight_bits: int = 4
    # 'auto' selects the method per layer (see select_layer_encoding); or force
    # 'kmeans' / 'linear_per_channel' / 'linear_per_tensor' for the whole model.
    weight_method: str = "auto"
    kmeans_init: str = "linear"
    # Rate penalty for entropy-constrained weight sharing. 0 is plain k-means.
    # Positive values trade reconstruction error for a lower-entropy code stream,
    # which the Huffman stage converts into real bits. See
    # weight_share.entropy_constrained_kmeans for why this is the right
    # objective: the pipeline pays for code *entropy*, not code count, and plain
    # k-means maximises entropy by construction.
    ecsq_lambda: float = 0.0
    # A candidate method is admissible if its reconstruction MSE is within this
    # factor of the best any method achieves for that layer; among admissible
    # candidates the cheapest in bits wins.
    method_mse_tolerance: float = 1.25
    # Exceptions: layers given their own (higher) bit width.
    depthwise_bits: Optional[int] = 8
    edge_bits: int = 8                     # stem convolution and classifier
    bn_bits: int = 8

    # ---- pruning
    sparsity: float = 0.0
    prune_scope: str = "global"            # 'global' | 'layer'
    max_layer_sparsity: float = 0.95

    # ---- entropy coding
    use_huffman: bool = True

    # ---- activations
    activation_bits: int = 8
    act_percentile: Optional[float] = 99.99
    quantize_activations: bool = True

    # ---- batchnorm
    fold_bn: bool = False

    def describe(self) -> str:
        return (f"w{self.weight_bits}({self.weight_method}) a{self.activation_bits} "
                f"sp{self.sparsity:.2f} dw{self.depthwise_bits} bn{self.bn_bits} "
                f"huff={'Y' if self.use_huffman else 'N'}")


# --------------------------------------------------------------------------- #
# Activation quantization: attach, calibrate, enable
# --------------------------------------------------------------------------- #
def attach_activation_quantizers(model: nn.Module, cfg: CompressionConfig
                                 ) -> Tuple[Dict[str, ActivationQuantizer], List]:
    """Insert an ActivationQuantizer at every activation site, via forward hooks.

    A forward hook that returns a value replaces the module's output, so the
    quantizers can be inserted without touching the model definition. This keeps
    the model file free of any compression code.

    Site selection and scheme:
      * ConvBNReLU outputs are post-ReLU6, hence one-sided in [0, 6]; an
        asymmetric quantizer spends all 2^b codes on the range that occurs,
        whereas a symmetric one would waste half of them on negative values that
        never appear - effectively a free extra bit.
      * InvertedResidual outputs come from a linear bottleneck (no activation
        after the projection) and are signed and roughly zero-mean, so a
        symmetric quantizer fits. Hooking the block rather than the projection
        convolution also avoids double-counting: in a residual block the summed
        output is what is materialised.
    """
    quantizers: Dict[str, ActivationQuantizer] = {}
    handles = []
    device = next(model.parameters()).device

    def make_hook(q: ActivationQuantizer):
        def hook(module, inputs, output):
            return q(output)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, ConvBNReLU):
            scheme = "asymmetric"
        elif isinstance(module, InvertedResidual):
            scheme = "symmetric"
        else:
            continue
        q = ActivationQuantizer(
            ActQuantConfig(num_bits=cfg.activation_bits, scheme=scheme,
                           percentile=cfg.act_percentile),
            name=name,
        ).to(device)
        quantizers[name] = q
        handles.append(module.register_forward_hook(make_hook(q)))

    return quantizers, handles


@torch.no_grad()
def calibrate_activations(model: nn.Module, quantizers: Dict[str, ActivationQuantizer],
                          calib_loader, device: torch.device,
                          max_batches: int = 8) -> None:
    """Observe activation ranges on held-out *training* data, then freeze.

    Calibrating on the test set would leak evaluation data into the compression
    parameters and inflate the reported post-compression accuracy, so the
    calibration split comes from the training set (see data.py).
    """
    for q in quantizers.values():
        q.calibrating, q.enabled = True, False
    model.eval().to(device)

    for i, (images, _) in enumerate(calib_loader):
        if i >= max_batches:
            break
        model(images.to(device, non_blocking=True))

    for q in quantizers.values():
        q.freeze()
        q.enabled = True


def measure_activation_cost(quantizers: Dict[str, ActivationQuantizer],
                            model: nn.Module, device: torch.device,
                            input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> ActivationCost:
    """Record the size of every quantized activation tensor for one inference.

    How the activations are measured (the assignment asks this explicitly):
    a single image is pushed through the network and the output tensor of every
    quantization site is recorded. The reported ratio is the sum over all such
    tensors of 32 bits/element (fp32) divided by the sum of b bits/element
    (quantized) - i.e. the reduction in activation *traffic* over one inference.
    The peak single-tensor figure is reported alongside, since that is what
    bounds the on-chip buffer.
    """
    shapes: Dict[str, Tuple[int, ...]] = {}

    def rec(nm):
        def hook(module, inputs, output):
            shapes[nm] = tuple(output.shape)
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in quantizers:
            handles.append(module.register_forward_hook(rec(name)))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape, device=device))
    for h in handles:
        h.remove()
    if was:
        model.train()

    sites = []
    for name, q in quantizers.items():
        shape = shapes.get(name)
        if shape is None:
            continue
        numel = 1
        for s in shape[1:]:            # exclude batch dimension
            numel *= s
        sites.append({"name": name, "shape": tuple(shape[1:]), "numel": numel,
                      "bits": q.cfg.num_bits, "scheme": q.cfg.scheme})
    return ActivationCost(sites=sites)


# --------------------------------------------------------------------------- #
# Weight compression
# --------------------------------------------------------------------------- #
def _encode_layer(codes: torch.Tensor, mask: torch.Tensor, num_bits: int,
                  use_huffman: bool, alphabet_size: int, zero_code: int = 0) -> Dict:
    """Pick the cheapest encoding of one layer's quantized codes.

    Searches jointly over
        - the sparse layout (dense / bitmap / relative index at five delta
          widths), and
        - whether to Huffman-code the value stream and the delta stream.

    The two interact: entropy coding shrinks the value stream but not the
    bitmap, which moves the layout crossover. Choosing them jointly per layer is
    therefore not the same as choosing them independently.

    Admissibility rules (both are correctness constraints, not optimisations):
      * dense is only offered when nothing is pruned - the code at a pruned
        position is meaningless and would not decode back to zero.
      * bitmap and relative index are only offered when something IS pruned; with
        a full mask a bitmap is strictly dense plus N wasted bits, and a relative
        index is dense plus a delta per weight.

    Costing runs off symbol histograms rather than materialised streams, since
    the search visits seven layouts per layer across 53 layers.
    """
    no_pruning = bool(mask.all())
    candidates: List[Dict] = []

    def stream_cost(counts: Dict[int, int], n_symbols: int, fixed_bits: int,
                    alpha: int) -> Tuple[int, bool]:
        """Cost of one stream, entropy-coded only where that genuinely wins.

        The canonical Huffman code-length table costs alpha * 5 bits whatever the
        stream length, so on short streams the table exceeds the saving and the
        fixed-width stream is kept instead.
        """
        raw = n_symbols * fixed_bits
        if not use_huffman or n_symbols == 0:
            return raw, False
        r = huffman_result_from_counts(counts, fixed_bits, alpha)
        return (r.total_bits, True) if r.total_bits < raw else (raw, False)

    flat_codes = codes.flatten().long()

    if no_pruning:
        hist = torch.bincount(flat_codes, minlength=alphabet_size)
        counts = {i: int(c) for i, c in enumerate(hist.tolist()) if c}
        v_bits, v_huff = stream_cost(counts, codes.numel(), num_bits, alphabet_size)
        candidates.append({"encoding": "dense", "value_bits": v_bits, "index_bits": 0,
                           "huffman": v_huff, "nnz": codes.numel()})
    else:
        # bitmap: one presence bit per weight, then codes for the survivors only
        kept = flat_codes[mask.flatten()]
        hist = torch.bincount(kept, minlength=alphabet_size)
        counts = {i: int(c) for i, c in enumerate(hist.tolist()) if c}
        v_bits, v_huff = stream_cost(counts, kept.numel(), num_bits, alphabet_size)
        candidates.append({"encoding": "bitmap", "value_bits": v_bits,
                           "index_bits": mask.numel(), "huffman": v_huff,
                           "nnz": int(kept.numel())})

        # relative index at several delta widths. The best width tracks the
        # achieved sparsity: too narrow and fillers multiply (at 98% sparsity a
        # 4-bit delta emits three fillers per real value), too wide and every
        # entry overpays.
        for ib in (3, 4, 5, 6, 8):
            st = relative_index_stats(mask, codes, ib, alphabet_size, zero_code=zero_code)
            n_entries = st["num_entries"]
            v_bits, v_huff = stream_cost(st["value_counts"], n_entries,
                                         num_bits, alphabet_size)
            d_bits, d_huff = stream_cost(st["delta_counts"], n_entries, ib, 2 ** ib)
            candidates.append({"encoding": f"rel{ib}", "value_bits": v_bits,
                               "index_bits": d_bits, "huffman": v_huff or d_huff,
                               "nnz": st["nnz"]})

    return min(candidates, key=lambda c: c["value_bits"] + c["index_bits"])


@torch.no_grad()
def _quantize_layer(w: torch.Tensor, mask: torch.Tensor, bits: int, method: str,
                    cfg: "CompressionConfig") -> Dict:
    """Quantize one weight tensor by one method, returning codes and cost pieces.

    Methods:
      'kmeans'             non-uniform codebook shared by the whole layer.
                           Metadata: 2^b fp32 centroids.
      'linear_per_channel' uniform grid, one fp32 scale per output channel.
                           Metadata: C_out * 32 bits.
      'linear_per_tensor'  uniform grid, a single fp32 scale for the layer.
                           Metadata: 32 bits.

    Metadata is the axis on which these differ sharply, and which one wins is
    layer-dependent: for a 307,200-weight pointwise convolution the k-means
    codebook (512 bits at b=4) is 20x cheaper than per-channel scales (10,240
    bits), but for the 864-weight stem at b=8 the codebook costs 8,192 bits -
    more than the 6,912 bits of values it indexes.
    """
    if method in ("kmeans", "ecsq"):
        lam = cfg.ecsq_lambda if method == "ecsq" else 0.0
        res = share_weights(w, bits, mask=mask, init=cfg.kmeans_init, lam=lam)
        return {"method": method, "codes": res.indices, "recon": res.reconstructed,
                "metadata_bits": res.codebook_bits, "alphabet": 2 ** bits,
                "zero_code": 0,
                "note": f"codebook {2**bits}x32b" + (f", lam={lam:g}" if lam else "")}

    gran = "per_channel" if method == "linear_per_channel" else "per_tensor"
    scale, zp = compute_qparams(w, bits, "symmetric", gran, channel_dim=0)
    q = quantize(w, scale, zp, bits, "symmetric", channel_dim=0)
    recon = dequantize(q, scale, zp, channel_dim=0) * mask
    # Shift signed codes to non-negative so the Huffman alphabet is dense.
    qmax = 2 ** (bits - 1) - 1
    codes = torch.where(mask, q + qmax, torch.full_like(q, float(qmax)))
    n_scales = scale.numel() if scale.ndim else 1
    return {"method": method, "codes": codes, "recon": recon,
            # Symmetric quantization has zero_point == 0 everywhere, so only the
            # scales need storing.
            "metadata_bits": n_scales * 32, "alphabet": 2 ** bits,
            "zero_code": qmax, "note": f"{n_scales} fp32 scale(s)"}


@torch.no_grad()
def select_layer_encoding(w: torch.Tensor, mask: torch.Tensor, bits: int,
                          cfg: "CompressionConfig") -> Dict:
    """Choose the quantization method and sparse layout for one layer.

    Selection rule: among the candidate methods, take the one with the fewest
    total bits *subject to* its reconstruction error being within
    `cfg.method_mse_tolerance` of the best error achieved by any candidate.

    Choosing by bits alone would be wrong. At b=4 on a large pointwise
    convolution the three methods differ by ~10 kbit of metadata against ~1.2
    Mbit of values, so a pure bit criterion is decided by noise in the entropy
    coder and would happily pick per-tensor linear - which has several times the
    distortion. Gating on error first, then minimising bits, picks k-means where
    it genuinely wins and switches to linear only where the codebook has stopped
    paying for itself.
    """
    if cfg.weight_method != "auto":
        cand = _quantize_layer(w, mask, bits, cfg.weight_method, cfg)
        cand["mse"] = float(((w - cand["recon"]) ** 2).mean())
        cand["enc"] = _encode_layer(cand["codes"].long(), mask, bits, cfg.use_huffman,
                                    cand["alphabet"], zero_code=cand["zero_code"])
        return cand

    candidates = []
    pool = ("ecsq", "linear_per_channel", "linear_per_tensor") if cfg.ecsq_lambda > 0 \
        else ("kmeans", "linear_per_channel", "linear_per_tensor")
    for method in pool:
        c = _quantize_layer(w, mask, bits, method, cfg)
        c["mse"] = float(((w - c["recon"]) ** 2).mean())
        c["enc"] = _encode_layer(c["codes"].long(), mask, bits, cfg.use_huffman,
                                 c["alphabet"], zero_code=c["zero_code"])
        c["total_bits"] = c["enc"]["value_bits"] + c["enc"]["index_bits"] + c["metadata_bits"]
        candidates.append(c)

    best_mse = min(c["mse"] for c in candidates)
    admissible = [c for c in candidates
                  if c["mse"] <= best_mse * cfg.method_mse_tolerance]
    return min(admissible, key=lambda c: c["total_bits"])


@torch.no_grad()
def compress_weights(model: nn.Module, cfg: CompressionConfig,
                     pruner: Optional[Pruner] = None) -> ModelCost:
    """Quantize (and optionally prune) every weight tensor, in place.

    The model's weights are replaced by their reconstructed (dequantized)
    values, so the returned model can be evaluated with ordinary float kernels
    while producing exactly the accuracy an integer implementation would.

    Returns the itemised storage cost of the result.
    """
    cost = ModelCost()
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    first_name, last_name = convs[0][0], convs[-1][0]

    for name, module in convs:
        w = module.weight.data
        is_depthwise = isinstance(module, nn.Conv2d) and module.groups > 1
        is_edge = name in (first_name, last_name)

        # ---- layer policy: bit width and whether pruning applies
        if is_edge:
            bits = cfg.edge_bits
            kind = "stem" if name == first_name else "classifier"
            prunable = False
        elif is_depthwise:
            bits = cfg.depthwise_bits if cfg.depthwise_bits is not None else cfg.weight_bits
            kind = "depthwise"
            prunable = False
        else:
            bits = cfg.weight_bits
            kind = "pointwise"
            prunable = True

        # ---- pruning mask
        if prunable and pruner is not None and name in pruner.masks:
            mask = pruner.masks[name]
        else:
            mask = torch.ones_like(w, dtype=torch.bool)

        # ---- quantize: pick method (and sparse layout) for this layer
        sel = select_layer_encoding(w, mask, bits, cfg)
        module.weight.data = sel["recon"].to(w.dtype)
        enc = sel["enc"]

        cost.layers.append(LayerCost(
            name=name, kind=kind, numel=w.numel(), nnz=enc["nnz"], num_bits=bits,
            value_bits=enc["value_bits"], index_bits=enc["index_bits"],
            metadata_bits=sel["metadata_bits"], encoding=enc["encoding"],
            huffman=enc["huffman"], notes=f"{sel['method']}, {sel['note']}",
        ))

        # ---- biases, where present, are small and kept at fp16
        if getattr(module, "bias", None) is not None:
            cost.layers.append(LayerCost(
                name=name + ".bias", kind=kind, numel=module.bias.numel(),
                nnz=module.bias.numel(), num_bits=16,
                value_bits=module.bias.numel() * 16, encoding="dense",
                notes="fp16 bias"))

    # ---- BatchNorm
    cost.layers.extend(batchnorm_cost(model, num_bits=cfg.bn_bits, folded=cfg.fold_bn))
    return cost


@torch.no_grad()
def quantize_batchnorm(model: nn.Module, num_bits: int) -> None:
    """Quantize BatchNorm parameters and buffers in place.

    Charged in `sizing.batchnorm_cost`; applied here so the evaluated accuracy
    reflects the storage we claim. Each of gamma, beta, running_mean and
    running_var is quantized per tensor (they are 1-D vectors, so per-channel
    granularity would mean one scale per value and save nothing).
    """
    from .quantize import fake_quantize
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for tensor, scheme in ((m.weight, "symmetric"), (m.bias, "symmetric"),
                                   (m.running_mean, "symmetric"),
                                   (m.running_var, "asymmetric")):
                if tensor is None:
                    continue
                tensor.data = fake_quantize(tensor.data, num_bits, scheme, "per_tensor")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class CompressionResult:
    """Everything one pipeline run produces."""

    config: CompressionConfig
    model_cost: ModelCost
    activation_cost: Optional[ActivationCost]
    fp32_baseline: Dict[str, int]
    summary: Dict[str, float]
    accuracy: Dict[str, float] = field(default_factory=dict)
    baseline_accuracy: Dict[str, float] = field(default_factory=dict)


def compress(model: nn.Module, cfg: CompressionConfig, calib_loader,
             device: torch.device, calib_batches: int = 8
             ) -> Tuple[nn.Module, CompressionResult]:
    """Run the full pipeline on a copy of `model`.

    The input model is never modified; a deep copy is compressed and returned,
    so the caller retains an untouched fp32 reference for the accuracy delta.

    Order of operations matters:
      1. baseline size is measured on the pristine model,
      2. pruning masks are computed from the *trained* weights,
      3. weights are quantized with the mask already known, so that pruned
         weights never influence the codebook,
      4. activation quantizers are attached and calibrated on the *compressed*
         weights, since that is the network that will actually run.
    """
    work = copy.deepcopy(model).to(device)
    baseline = fp32_model_bits(work, include_bn_buffers=True)

    # stage 1: pruning
    pruner = None
    if cfg.sparsity > 0:
        pruner = Pruner(work, PruneConfig(sparsity=cfg.sparsity, scope=cfg.prune_scope,
                                          max_layer_sparsity=cfg.max_layer_sparsity))
        pruner.compute_masks()
        pruner.apply()

    # stages 2 and 3: weight sharing / linear quantization, then entropy coding
    model_cost = compress_weights(work, cfg, pruner)
    if not cfg.fold_bn:
        quantize_batchnorm(work, cfg.bn_bits)

    # activations
    act_cost = None
    if cfg.quantize_activations:
        quantizers, _handles = attach_activation_quantizers(work, cfg)
        calibrate_activations(work, quantizers, calib_loader, device, calib_batches)
        act_cost = measure_activation_cost(quantizers, work, device)

    summary = summarise(model_cost, act_cost, baseline)
    return work, CompressionResult(config=cfg, model_cost=model_cost,
                                   activation_cost=act_cost, fp32_baseline=baseline,
                                   summary=summary)

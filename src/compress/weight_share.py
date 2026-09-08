"""K-means weight sharing (trained quantization), written from scratch.

Stage 2 of the Deep Compression pipeline. No clustering library is used:
Lloyd's algorithm is implemented directly below, specialised to the 1-D case,
which lets it run on the GPU over a whole layer at once.

The idea
--------
Instead of placing quantization levels on a uniform grid, cluster the weights of
a layer into 2^b groups and store, per weight, only the b-bit index of its
cluster. The 2^b cluster centroids are stored once per layer as fp32.

    storage(layer) = nnz * b  +  2^b * 32   bits

Compared with per-channel linear quantization the difference is where the
metadata goes. Linear quantization needs one fp32 scale per output channel
(C_out * 32 bits); weight sharing needs one fp32 codebook per layer
(2^b * 32 bits). For a 320x960 pointwise convolution at b=4 that is 10,240 bits
versus 512 bits - a 20x difference in metadata, in favour of weight sharing.

The second advantage is that the codebook is non-uniform, so it can place levels
densely where the weights actually are (near zero) and sparsely in the tails.
A uniform grid must spend the same resolution everywhere.

The cost is that inference needs a lookup table rather than integer arithmetic,
so weight sharing compresses storage without directly enabling integer kernels.
Both options are implemented, and `pipeline.py` selects between them by
configuration so the trade-off can be measured rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

InitMethod = Literal["linear", "density", "random"]


@dataclass
class WeightShareResult:
    """Outcome of clustering one weight tensor."""

    centroids: torch.Tensor       # (k,) fp32 codebook
    indices: torch.Tensor         # same shape as the weight, int64 cluster ids
    reconstructed: torch.Tensor   # centroids[indices], the dequantized weight
    num_bits: int
    inertia: float                # sum of squared errors, the k-means objective
    iterations: int

    @property
    def codebook_bits(self) -> int:
        """fp32 codebook cost. This is metadata and is charged in `sizing.py`."""
        return self.centroids.numel() * 32


def _init_centroids(values: torch.Tensor, k: int, method: InitMethod) -> torch.Tensor:
    """Initialise k centroids over a 1-D tensor of weights.

    'linear' spaces centroids uniformly between min and max. Deep Compression
    found this to be the best of the three: density-based and random
    initialisation both concentrate centroids near zero, where the weights are,
    and therefore under-represent the large-magnitude weights - which are
    precisely the weights that matter most to the output. Linear initialisation
    deliberately spends centroids on the tails.
    """
    lo, hi = values.min(), values.max()
    if method == "linear":
        return torch.linspace(lo.item(), hi.item(), k, device=values.device, dtype=torch.float32)
    if method == "density":
        # Quantiles of the empirical distribution: many centroids near zero.
        qs = torch.linspace(0, 1, k, device=values.device)
        return torch.quantile(values.float(), qs)
    perm = torch.randperm(values.numel(), device=values.device)[:k]
    return values.flatten()[perm].float().clone()


def kmeans_1d(values: torch.Tensor, num_bits: int, init: InitMethod = "linear",
              max_iters: int = 50, tol: float = 1e-7, k: Optional[int] = None,
              ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    """Lloyd's algorithm on a 1-D tensor, k = 2^num_bits unless `k` is given.

    Each iteration:
      1. Assignment - every value takes the index of its nearest centroid.
         Done as a single (N, k) distance matrix on the GPU; for the largest
         MobileNet layer (307k weights, k=16) that is a 4.9M-element temporary,
         which is comfortable.
      2. Update - every centroid moves to the mean of the values assigned to it.
         Empty clusters keep their previous position rather than collapsing to
         NaN, so k stays fixed and the codebook never develops holes.

    Converges when the largest centroid movement falls below `tol`. Lloyd's
    algorithm is guaranteed to decrease the inertia monotonically, so the loop
    always terminates.

    Returns:
        (centroids, assignments, inertia, iterations_run)
    """
    k = k if k is not None else 2 ** num_bits
    flat = values.flatten().float()

    # Fewer distinct values than clusters: use the distinct values directly.
    unique = torch.unique(flat)
    if unique.numel() <= k:
        centroids = torch.zeros(k, device=flat.device, dtype=torch.float32)
        centroids[: unique.numel()] = unique
        centroids[unique.numel():] = unique[-1] if unique.numel() else 0.0
        assign = torch.searchsorted(unique.contiguous(), flat.contiguous()).clamp(max=k - 1)
        recon = centroids[assign]
        return centroids, assign, float(((flat - recon) ** 2).sum()), 0

    centroids = _init_centroids(flat, k, init)

    assign = torch.zeros_like(flat, dtype=torch.long)
    iterations = 0
    for iterations in range(1, max_iters + 1):
        # --- 1. assignment step: every weight takes its nearest centroid.
        # unsqueeze turns (N,) and (k,) into (N,1) and (1,k); broadcasting then
        # produces an (N, k) matrix of every weight-to-centroid distance. For
        # the largest layer that is 307,200 x 16 = 4.9M entries, comfortable on
        # a GPU, and it assigns the whole layer in one operation.
        dist = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()   # (N, k)
        assign = dist.argmin(dim=1)                                 # (N,) in [0, k)

        # --- 2. update step: each centroid moves to the mean of its members.
        # index_add_ is a scatter-add: it walks `assign` and accumulates each
        # weight into its cluster's slot, giving per-cluster sums and counts
        # without a Python loop.
        new_centroids = centroids.clone()
        sums = torch.zeros_like(centroids).index_add_(0, assign, flat)
        counts = torch.zeros_like(centroids).index_add_(0, assign, torch.ones_like(flat))
        # A cluster nobody chose would divide by zero, so it keeps its previous
        # position. It stays in the codebook and may attract weights later,
        # which keeps k fixed and leaves no holes in the code alphabet.
        nonempty = counts > 0
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty]

        # Stop once no centroid moves meaningfully. Lloyd's algorithm can only
        # decrease the total squared error, so this loop always terminates.
        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if shift < tol:
            break

    recon = centroids[assign]
    inertia = float(((flat - recon) ** 2).sum())
    return centroids, assign, inertia, iterations


def share_weights(weight: torch.Tensor, num_bits: int,
                  mask: Optional[torch.Tensor] = None,
                  init: InitMethod = "linear",
                  max_iters: int = 50,
                  reserve_zero: Optional[bool] = None,
                  lam: float = 0.0) -> WeightShareResult:
    """Cluster a weight tensor into 2^num_bits shared values.

    Args:
        weight: the tensor to compress.
        num_bits: bits per stored index; k = 2^num_bits centroids.
        mask: optional pruning keep-mask. Pruned weights are excluded from the
            clustering entirely - including them would drag centroids toward
            zero and waste codebook entries representing values that are never
            stored, since the sparse encoding already records position.
        init: centroid initialisation, see `_init_centroids`.
        lam: rate penalty for entropy-constrained clustering. 0 gives plain
            k-means; positive values trade reconstruction error for a
            lower-entropy code stream, which the Huffman stage then exploits.
            See `entropy_constrained_kmeans`.
        reserve_zero: reserve code 0 for an exact 0.0 and cluster the live
            weights into 2^num_bits - 1 centroids instead of 2^num_bits.
            Defaults to True whenever a pruning mask is present.

            This is required for correctness, not merely tidiness. The
            relative-index encoder emits "filler" entries that carry a zero
            weight, so some code must decode to exactly 0.0. Ordinary k-means
            offers no such code - index 0 is the most-negative centroid - so
            fillers would reconstruct as large negative weights and the encoding
            would not round-trip. Reserving a zero slot, as Deep Compression
            does, makes the sparse encodings decodable.

    Returns:
        WeightShareResult, where `reconstructed` already has the pruned
        positions re-zeroed so it can be loaded straight back into the model.
    """
    if mask is None:
        mask = torch.ones_like(weight, dtype=torch.bool)
    if reserve_zero is None:
        reserve_zero = not bool(mask.all())

    live = weight[mask]
    if live.numel() == 0:
        return WeightShareResult(
            centroids=torch.zeros(2 ** num_bits, device=weight.device),
            indices=torch.zeros_like(weight, dtype=torch.long),
            reconstructed=torch.zeros_like(weight),
            num_bits=num_bits, inertia=0.0, iterations=0,
        )

    fit = (lambda v, kk: entropy_constrained_kmeans(v, num_bits, lam, init, max_iters,
                                                    k=kk, normalize_lambda=True)) \
        if lam > 0 else (lambda v, kk: kmeans_1d(v, num_bits, init, max_iters, k=kk))

    if reserve_zero:
        # Cluster into 2^b - 1 groups, then prepend an exact zero as code 0.
        centroids, assign, inertia, iters = fit(live, 2 ** num_bits - 1)
        centroids = torch.cat([torch.zeros(1, device=centroids.device,
                                           dtype=centroids.dtype), centroids])
        assign = assign + 1
    else:
        centroids, assign, inertia, iters = fit(live, 2 ** num_bits)

    indices = torch.zeros_like(weight, dtype=torch.long)
    indices[mask] = assign
    reconstructed = torch.zeros_like(weight)
    reconstructed[mask] = centroids[assign].to(weight.dtype)

    return WeightShareResult(centroids=centroids, indices=indices,
                             reconstructed=reconstructed, num_bits=num_bits,
                             inertia=inertia, iterations=iters)


@torch.no_grad()
def update_centroids_from_gradients(weight: torch.Tensor, grad: torch.Tensor,
                                    indices: torch.Tensor, centroids: torch.Tensor,
                                    mask: torch.Tensor, lr: float) -> torch.Tensor:
    """One step of Deep Compression's *trained* quantization.

    All weights sharing a centroid must keep sharing it, so their individual
    gradients are summed into a single update for that centroid:

        centroid_j <- centroid_j - lr * sum_{i : index_i = j} grad_i

    This is what makes the codebook trainable: the shared values move to reduce
    the loss, while the assignment of weights to clusters stays fixed. It
    recovers most of the accuracy lost to clustering, which is why Deep
    Compression can reach 4 bits with negligible degradation where plain
    post-training clustering cannot.
    """
    # Weights sharing a centroid must keep sharing it, so they cannot move
    # independently. Their gradients are summed into one update for the shared
    # value; index_add_ does the grouping by cluster index.
    grad_sums = torch.zeros_like(centroids)
    grad_sums.index_add_(0, indices[mask].flatten(), grad[mask].flatten().float())
    # An ordinary SGD step, but on 2^b shared values instead of N weights. Which
    # weight belongs to which cluster stays fixed; only the values move.
    return centroids - lr * grad_sums


def entropy_constrained_kmeans(values: torch.Tensor, num_bits: int, lam: float,
                               init: InitMethod = "linear", max_iters: int = 50,
                               tol: float = 1e-7, k: Optional[int] = None,
                               normalize_lambda: bool = True,
                               ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    """Entropy-constrained scalar quantization (Chou, Lookabaugh & Gray, 1989).

    Plain k-means minimises distortion at a fixed *number* of codes. But the
    storage cost of this pipeline is not the number of codes - it is the
    *entropy* of the code stream, because stage 3 entropy-codes it. Those are
    different objectives, and optimising the first actively harms the second:
    minimising inertia spreads the weights evenly across all 2^b clusters, which
    drives the code distribution toward uniform, and a uniform distribution is
    exactly the one Huffman cannot compress.

    Measured on a trained MobileNet-v2 layer at b=4, k-means produces a code
    stream of entropy 2.94 bits while per-tensor linear quantization produces one
    of 0.94 bits - so linear wins on total storage by 2.3x despite 13x the
    reconstruction error.

    The fix is to put the rate term inside the clustering objective. Assign each
    weight to the cluster minimising the Lagrangian

        J(x, j) = (x - c_j)^2  +  lam * (-log2 p_j)

    where p_j is the current occupancy of cluster j, so -log2 p_j is the length
    the entropy coder would give that symbol. Frequently-used clusters become
    cheaper to select and attract more weights; rarely-used ones become
    expensive and empty out. The result is a codebook matched to a *low-entropy*
    code distribution.

    lam sweeps the rate-distortion curve:
        lam = 0     recovers plain k-means (minimum distortion, maximum rate)
        lam -> inf  collapses to a single code (zero rate, maximum distortion)
    Intermediate values trade the two, and the useful setting is found by
    measurement, not derivation.

    Units matter here, and getting them wrong is catastrophic rather than
    merely suboptimal. The distortion term is in weight^2 and the rate term is
    in bits, so a raw lam carries units of weight^2 per bit. MobileNet-v2 layers
    differ in weight scale by more than an order of magnitude, so one global raw
    lam is a mild penalty in some layers and a total collapse in others: at
    lam = 3e-5 the codebook of several layers pruned itself to 1-3 codes and
    top-1 fell to 10.16%, i.e. random guessing.

    With `normalize_lambda` the penalty is scaled by the layer's mean square
    weight, E[w^2], making lam dimensionless and the trade-off scale-invariant
    across layers. A useful range is then roughly 1e-4 to 1e-1 for every layer
    alike, and the collapse cliff disappears.

    Clusters that empty out are retained in the codebook but never selected
    again (their occupancy floor keeps the log finite). This is deliberate: it
    is how the method prunes its own alphabet, and the unused entries are still
    charged for in the codebook cost.

    Returns:
        (centroids, assignments, inertia, iterations_run)
    """
    k = k if k is not None else 2 ** num_bits
    flat = values.flatten().float()
    n = flat.numel()

    unique = torch.unique(flat)
    if unique.numel() <= k:
        return kmeans_1d(values, num_bits, init, max_iters, tol, k=k)

    # Make lam dimensionless: distortion is in weight^2, rate is in bits, so the
    # penalty must be expressed relative to the layer's own signal power.
    lam_eff = lam * float((flat ** 2).mean()) if normalize_lambda else lam

    centroids = _init_centroids(flat, k, init)
    # Start from a uniform occupancy prior, i.e. no rate preference yet.
    probs = torch.full((k,), 1.0 / k, device=flat.device, dtype=torch.float32)
    # Occupancy floor: prevents -log2(0) = inf and keeps the cost matrix finite.
    floor = 1.0 / (2.0 * n)

    assign = torch.zeros_like(flat, dtype=torch.long)
    iterations = 0
    for iterations in range(1, max_iters + 1):
        # --- assignment step, with the rate penalty folded in
        # -log2(p) is the code length an entropy coder gives a symbol used a
        # fraction p of the time: a cluster holding 50% of the weights costs
        # 1 bit to name, one holding 1% costs about 6.6 bits.
        rate = -torch.log2(probs.clamp(min=floor))                    # (k,) bits
        dist = (flat.unsqueeze(1) - centroids.unsqueeze(0)) ** 2      # (N, k)
        # Assign by distortion PLUS rate, not distortion alone. A popular
        # cluster is cheap to name and attracts more weights; an unpopular one
        # is expensive and empties out. That feedback pushes the code
        # distribution away from uniform, which is exactly what plain k-means
        # does not do, and why plain k-means works against the Huffman stage.
        assign = (dist + lam_eff * rate.unsqueeze(0)).argmin(dim=1)

        # --- update step: centroids to cluster means, probabilities to occupancy
        sums = torch.zeros_like(centroids).index_add_(0, assign, flat)
        counts = torch.zeros_like(centroids).index_add_(0, assign, torch.ones_like(flat))
        new_centroids = centroids.clone()
        nonempty = counts > 0
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty]
        probs = counts / n

        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if shift < tol:
            break

    recon = centroids[assign]
    inertia = float(((flat - recon) ** 2).sum())
    return centroids, assign, inertia, iterations

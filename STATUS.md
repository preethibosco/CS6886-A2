# CS6886 Assignment 2 - Status

**Due:** Monday 8 September 2026.  **Target:** MobileNet-v2 / CIFAR-10, train +
compress.  Q3 and Q4 are relatively graded on compression ratio and accuracy.

## Where things stand

| Assignment part | Status |
|---|---|
| Q1a transforms | done - `src/data.py`, printed spec for the report |
| Q1b architecture + training config | done - `src/models/mobilenetv2.py`, `src/train.py` |
| Q1c accuracy + curves + failure modes | done - **95.17%** top-1, figures in `results/figures/` |
| Q2a configurable compression method | done - 4 hand-written stages |
| Q2b applied to MobileNet-v2, exceptions | done - layer policy in `pipeline.py` docstring |
| Q2c storage overheads | done - `src/compress/sizing.py`, itemised |
| Q3 sweep + wandb parallel coordinates | done - 60 configurations logged to wandb |
| Q4 final numbers | best so far **31.40x at 94.24%** (w3/a8 sp0.80); optimisation campaign running |
| Q5a modular codebase | done |
| Q5b README | done - commands, versions, seeds |
| Q5c GitHub repo | local `main` committed and ready; remote is yours to create |

## Baseline training

300 epochs, SGD + cosine, seed 42. Live at
https://wandb.ai/Preethi_manifold/cs6886-a2-mobilenetv2
Finished: **95.17%** test top-1 at epoch 293, 45.8 min. Train 99.46%, so a
4.32 pt generalisation gap. Failure modes are semantically concentrated: cat
(88.2%) and dog (91.6%) form one reciprocal confusion pair; the eight rigid-body
classes are all >=94%.

## The compression pipeline

Four hand-written stages (no compression library - enforced by an AST scan):

1. **Pruning** (`prune.py`) - fine-grained magnitude, global threshold with a
   per-layer cap. Sparse layout chosen per layer from {bitmap, relative index at
   3/4/5/6/8-bit deltas}, each with a decoder.
2. **Weight sharing** (`weight_share.py`) - Lloyd's algorithm, plus
   entropy-constrained clustering (ECSQ) with a rate penalty.
3. **Entropy coding** (`huffman.py`) - canonical Huffman, code table charged.
4. **Activation quantization** (`quantize.py`) - asymmetric post-ReLU6,
   symmetric post-linear-bottleneck, calibrated on held-out *training* data.

Plus `qat.py` (prune-and-retrain + STE fine-tuning) and `sizing.py` (accounting).

## Key findings so far

* **k-means fights Huffman.** k-means minimises inertia, which spreads weights
  evenly across clusters and *maximises* code entropy - exactly what Huffman
  cannot compress. Measured on `features.17.conv.2` at b=4: k-means gives code
  entropy 2.94 and 920 kbit; per-tensor linear gives entropy 0.94 and 401 kbit,
  despite 13x worse reconstruction error. The pipeline pays for entropy, not for
  code count.
* **Fix: entropy-constrained quantization** with the rate term inside the
  clustering objective. Gives a tunable curve; at matched distortion it beats
  per-tensor linear by ~15% on the head layer.
* **Delta width must be swept, not inherited.** Deep Compression's 4-bit
  relative index wastes 60% of the budget at 98% sparsity (fillers outnumber
  real values 3:1). Per-layer width search recovers it.
* **Metadata dominates after compression.** BatchNorm buffers are 1.5% of the
  fp32 model but a double-digit share of the compressed one. A 256-entry
  codebook (8,192 bits) exceeds the stem layer's 864 weights.

## Verification

`scripts/conformance.py` - 21 checks, architecture -> implementation ->
accounting. The decisive one decodes the claimed bitstream and compares it
against the weights the accuracy number came from. Added after two silent bugs
(undecodable dense encoding under pruning; filler entries with no zero code)
reached execution while printing plausible ratios. See `LEARNINGS.md`.

## Compression results (QAT, gradient clipping on)

| Config | Ratio | Size | PTQ | QAT | vs fp32 |
|---|---|---|---|---|---|
| w4/a8 sp0.80 | 23.31x | 0.372 MB | 45.18 | 94.73 | -0.44 |
| w4/a8 sp0.95 | 31.26x | 0.277 MB | 10.00 | 92.47 | -2.70 |
| w3/a8 sp0.70 | 31.59x | 0.274 MB | 32.42 | 94.20 | -0.97 |
| **w3/a8 sp0.80** | **31.40x** | **0.276 MB** | 27.12 | **94.24** | **-0.93** |
| w3/a8 sp0.90 | 32.23x | 0.269 MB | 15.41 | 93.63 | -1.54 |

Two further results from these:

* **3-bit at moderate sparsity beats 4-bit at heavy sparsity.** w3/sp0.80 and
  w4/sp0.95 reach the same ratio (31.4x vs 31.3x) but differ by 1.77 points of
  top-1. 3-bit per-tensor quantization is itself an implicit pruner - 88.3% of
  weights round to zero at sparsity 0 - so explicit pruning re-harvests
  redundancy that has already been removed, while still paying index metadata.
* **At 3 bits, explicit pruning buys almost nothing.** 70% -> 90% sparsity moves
  the ratio 31.59x -> 32.23x (+2%) and costs 0.6 points.

## Where the compressed bits sit (w3/a8 sp0.80)

| Component | Share of compressed model | Bits |
|---|---|---|
| pointwise | 63.4% | 3 |
| BatchNorm | 18.2% | 8 |
| depthwise | 15.4% | 8 |
| classifier | 2.9% | 8 |

Index metadata is 37.9% of all storage. The running optimisation campaign
attacks all three: drop pruning, and push the two 8-bit blocks that now dominate.

## Negative results (reported, not dropped)

* **ECSQ did not pay off end-to-end.** The per-layer entropy analysis was right -
  k-means maximises code entropy and fights Huffman - but entropy-constrained
  clustering lost to plain per-tensor linear on both ratio and accuracy, because
  it sacrifices the large-magnitude tail that dominates the output.
* **Unguarded QAT is a lottery.** Without gradient clipping, gradients reach
  8.3e5, one layer's weights diverge to 6.7e14 and the loss pins at ln(10)
  permanently. It is absorbing and batch-order dependent, so identical
  configurations succeed or fail run to run. Every pre-clipping number was
  re-run; the old "46.66x at 88.01%" was a partially destroyed model whose
  degenerate weights compressed well and classified badly.

## Next

1. Finish the optimisation campaign (pruning off; depthwise/BN below 8 bits).
2. Q4 final numbers table from the campaign JSON.
3. Report PDF.
4. GitHub remote (tree is committed locally).

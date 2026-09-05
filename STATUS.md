# CS6886 Assignment 2 - Status

**Due:** Monday 8 September 2026.  **Target:** MobileNet-v2 / CIFAR-10, train +
compress.  Q3 and Q4 are relatively graded on compression ratio and accuracy.

## Where things stand

| Assignment part | Status |
|---|---|
| Q1a transforms | done - `src/data.py`, printed spec for the report |
| Q1b architecture + training config | done - `src/models/mobilenetv2.py`, `src/train.py` |
| Q1c accuracy + curves + failure modes | training in progress; `scripts/plot_curves.py` ready |
| Q2a configurable compression method | done - 4 hand-written stages |
| Q2b applied to MobileNet-v2, exceptions | done - layer policy in `pipeline.py` docstring |
| Q2c storage overheads | done - `src/compress/sizing.py`, itemised |
| Q3 sweep + wandb parallel coordinates | `src/sweep.py` written, not yet run at scale |
| Q4 final numbers | pending - needs the finished baseline + QAT |
| Q5a modular codebase | done |
| Q5b README | **not written yet** |
| Q5c GitHub repo | **not created yet** |

## Baseline training

300 epochs, SGD + cosine, seed 42. Live at
https://wandb.ai/Preethi_manifold/cs6886-a2-mobilenetv2
Currently ~epoch 212, test top-1 92.35%.

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

## Next

1. Finish baseline (~epoch 300), generate curves.
2. Pick the rate-distortion operating point end-to-end (`sweep_lambda.py`).
3. QAT fine-tune the chosen configurations to recover accuracy.
4. Full sweep -> wandb parallel coordinates chart.
5. README + GitHub repo + report PDF.

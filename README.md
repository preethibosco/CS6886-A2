# CS6886 Assignment 2 — MobileNet-v2 on CIFAR-10, trained and compressed

Training MobileNet-v2 from scratch on CIFAR-10, then compressing it with a
four-stage pipeline written from scratch: **magnitude pruning → weight
sharing / linear quantization → canonical Huffman coding**, plus calibrated
**activation quantization**.

No compression API or library function is used anywhere. `torch.ao.quantization`,
`torch.nn.utils.prune`, `sklearn`, and the `zlib`/`gzip` family are all absent —
enforced automatically by an AST scan in `scripts/conformance.py`, not by
convention.

---

## 1. Environment

```bash
python -m venv venv
./venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
./venv/bin/pip install wandb matplotlib numpy
```

Exact versions used for every number in the report (`requirements.txt`):

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| numpy | 2.2.6 |
| wandb | 0.29.0 |
| matplotlib | 3.10.9 |
| Python | 3.10 |

Hardware: single NVIDIA RTX 5090 (`sm_120`, CUDA 13.1). The cu128 wheel is
required — earlier PyTorch builds ship no kernels for `sm_120`.

## 2. Seeds and reproducibility

Every entry point takes `--seed` (default **42**) and calls
`src.utils.seed_everything`, which seeds `random`, `numpy`, `torch` (CPU and all
CUDA devices) and `PYTHONHASHSEED`.

* **Training** runs with `cudnn.benchmark=True` for throughput, so it is
  seed-controlled but not bit-deterministic (CUDA atomics in the backward pass).
* **Every compression and evaluation script** passes `deterministic=True`, which
  sets `cudnn.deterministic=True`, disables benchmarking and sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. All reported compression numbers are
  therefore exactly reproducible.
* The **calibration split** is drawn with a dedicated `torch.Generator` seeded
  from the same value, so the activation ranges are identical across runs.

## 3. Reproducing the results

```bash
# 0. verify the implementation (21 checks: architecture -> encoders -> accounting)
./venv/bin/python scripts/conformance.py

# 1. train the baseline  (~46 min on one RTX 5090)
./venv/bin/python -m src.train --epochs 300 --seed 42 --wandb \
    --run-name baseline-mbv2-w1.0-300ep

# 2. training curves, per-class accuracy, confusion matrix  (Q1c)
./venv/bin/python scripts/plot_curves.py

# 3. compress at one configuration and print the full accounting
./venv/bin/python scripts/compress_eval.py \
    --weight-bits 4 --activation-bits 8 --sparsity 0.8 --full-table

# 4. the sweep behind the wandb Parallel Coordinates chart  (Q3)
./venv/bin/python -m src.sweep --weight-bits 2,3,4,6,8 \
    --activation-bits 2,4,6,8 --sparsities 0.0,0.5,0.8 --wandb

# 5. the reported configuration, with quantization-aware fine-tuning  (Q4)
./venv/bin/python -m src.sweep --quick --qat --qat-epochs 15
```

Supporting experiments (each answers one design question with a measurement):

```bash
./venv/bin/python scripts/inspect_model.py            # shapes, parameter budget
./venv/bin/python scripts/explain_config.py           # (t,c,n,s) -> channels, depth
./venv/bin/python scripts/layer_table.py              # per-layer weights + activations
./venv/bin/python scripts/test_prune_encoding.py      # sparse encoders, delta-width search
./venv/bin/python scripts/test_huffman.py             # entropy bound, when Huffman loses
./venv/bin/python scripts/test_entropy_interaction.py # k-means vs Huffman entropy
./venv/bin/python scripts/test_ecsq.py                # rate-distortion sweep
./venv/bin/python scripts/sweep_sparsity.py           # sparsity x method
./venv/bin/python scripts/sweep_lambda.py             # entropy-penalty operating point
./venv/bin/python scripts/sweep_method_tolerance.py   # per-layer method selection
```

## 4. Repository layout

```
src/
  data.py                  CIFAR-10 transforms + train/test/calibration loaders   (Q1a)
  models/mobilenetv2.py    MobileNet-v2, written from scratch, CIFAR stem         (Q1b)
  train.py                 baseline training loop                                 (Q1b)
  evaluate.py              top-1/top-5, per-class accuracy, confusion matrix
  sweep.py                 compression sweep + wandb logging                      (Q3)
  utils.py                 seeding, device, meters
  compress/
    quantize.py            linear quantization, STE, activation quantizers        (Q2a)
    prune.py               magnitude pruning + sparse index encoding              (Q2a)
    weight_share.py        k-means and entropy-constrained weight sharing         (Q2a)
    huffman.py             canonical Huffman coding                               (Q2a)
    qat.py                 prune-and-retrain and quantization-aware fine-tuning
    sizing.py              exact storage accounting incl. all overheads           (Q2c)
    pipeline.py            layer policy and orchestration                         (Q2b)
scripts/                   verification and experiment drivers
results/                   metrics, figures, sweep output
LEARNINGS.md               transferable rules, each traced to a real bug
STATUS.md                  current state against Q1-Q5
```

Training, evaluation and compression are separate import paths. Nothing in
`compress/` imports `train.py`, and `models/mobilenetv2.py` contains no
compression code — the activation quantizers attach through forward hooks.

## 5. Method summary

**Which layers are compressed, and the exceptions (Q2b).**

| Layers | Share of params | Treatment |
|---|---|---|
| Pointwise 1×1 convolutions | 94.99% | pruned + quantized to `weight_bits` |
| Depthwise 3×3 convolutions | 2.87% | quantized to 8 bits, **not pruned** |
| BatchNorm γ, β + running buffers | 1.53% (+1.53% buffers) | quantized to 8 bits |
| Classifier | 0.57% | quantized to 8 bits, not pruned |
| Stem convolution | 0.04% | quantized to 8 bits, not pruned |

Depthwise layers are exempt because each channel holds only 9 weights and does
no cross-channel mixing, so there is no redundancy to prune and no neighbouring
channel to absorb quantization error. They are 2.9% of the model, so protecting
them costs almost nothing in compression ratio.

**Storage overheads, all charged (Q2c).** Nothing a decoder needs is free:
quantized value codes (after Huffman where it pays), sparse position metadata
(bitmap or relative-index deltas), the k-means codebook or per-channel scales,
Huffman code-length tables, BatchNorm parameters **and** the
`running_mean`/`running_var` buffers that `model.parameters()` does not report.

**How activations are measured (Q4b).** One image is pushed through the network
and the output tensor of every quantization site is recorded. The reported ratio
is the sum over all such tensors of 32 bits/element divided by the sum of
`b` bits/element — the reduction in activation **traffic** over one inference.
The **peak** single-tensor figure is reported alongside, since that is what
bounds the on-chip buffer.

## 6. Verification

`scripts/conformance.py` runs 21 checks spanning architecture → implementation →
accounting. The decisive one (section G) takes the encoding the reported size
figure claims, **decodes it, and compares against the weights the accuracy
number was produced from**. It exists because two bugs previously reached
execution while printing entirely plausible compression ratios. See
`LEARNINGS.md`.

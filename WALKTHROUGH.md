# Code walkthrough

A reading order for the repository, written for someone meeting this code for
the first time. Each step says what to open, what it does, and the one idea that
makes the rest of the file make sense.

Every source file carries inline comments with tensor shapes and worked numeric
examples, so this document is a map rather than a substitute for reading them.

## Before anything else: run these three

They print the facts the whole design rests on.

```bash
./venv/bin/python scripts/inspect_model.py     # shapes and where the parameters live
./venv/bin/python scripts/explain_config.py    # how (t,c,n,s) becomes 53 layers
./venv/bin/python scripts/layer_table.py       # per-layer weights AND activations
```

The single most important number they print:

| Region | Weights | Activations (batch 1) |
|---|---|---|
| `features.0`-`features.4` | 23,936 (1.1%) | 815,104 (50%) |
| `features.15`-`features.18` | 1,638,400 (74%) | 122,880 (7.5%) |

Weights and activations sit in opposite ends of the network. That is why the
assignment asks for their compression ratios separately, and why the code treats
them as two different problems.

---

## Part 1: the network

### 1. `src/models/mobilenetv2.py`

The architecture, written out rather than imported.

Read in this order: `_make_divisible`, then `ConvBNReLU`, then
`InvertedResidual`, then the `MobileNetV2.__init__` loop.

The one idea: an inverted residual block goes **narrow, wide, narrow**. It
expands the channels with a 1x1 convolution, does the spatial mixing with a
cheap depthwise convolution while the tensor is wide, then projects back down.
The projection has BatchNorm but no activation, which is the "linear
bottleneck" the paper is named for.

Two consequences you will meet again:

* The 1x1 convolutions hold 95% of the parameters. Compression is about them.
* Post-ReLU6 tensors are one-sided in [0,6]; post-projection tensors are signed.
  Those two need different activation quantizers.

The CIFAR adaptation is in the `__init__` loop: two stride-2 steps are turned
into stride-1, because the stock schedule downsamples 32x and would collapse a
32x32 input to a 1x1 feature map.

### 2. `src/data.py`

Transforms and three loaders. The third loader is the interesting one: a
calibration split carved out of the **training** set, used to observe activation
ranges. Fitting those ranges on test data would leak the evaluation set into the
compression parameters.

### 3. `src/train.py`, `src/evaluate.py`

Standard training loop. Two details worth pausing on: weight decay is withheld
from BatchNorm and depthwise weights (`build_param_groups`), and the learning
rate follows warmup then cosine (`lr_at`).

`evaluate.py` is separate so that "accuracy before" and "accuracy after" come
from literally the same code path.

---

## Part 2: compression, in pipeline order

The pipeline is Deep Compression (Han et al., ICLR 2016) with an activation
stage added. Read the four stages in the order the data flows through them.

### 4. `src/compress/quantize.py` - the arithmetic

Start here; everything else uses it.

```
q  = clamp(round(x/s) + z, qmin, qmax)      quantize
x^ = s * (q - z)                            dequantize
```

Read `quant_bounds`, `compute_qparams`, `fake_quantize`, then `_RoundSTE`.

The one idea: **fake quantization**. Quantize and immediately dequantize, so the
value lands on the integer grid but stays a float tensor. Ordinary convolutions
then run on it, and the accuracy is exactly what an integer kernel would give.
No integer kernels are ever written.

`_RoundSTE` is how gradients survive `round()`, whose true derivative is zero
everywhere. It is the reason fine-tuning works at all.

### 5. `src/compress/prune.py` - stage 1

Two halves, and the second is the one people skip.

* Choosing what to remove: `magnitude_mask`, `global_masks`, `Pruner`.
* Choosing how to **store** what survives: `encode_bitmap`,
  `encode_relative_index`, and their decoders.

The one idea: a sparse tensor is only smaller if you can encode the positions of
the survivors cheaply. Bitmap costs one bit per weight whatever the sparsity;
relative indexing costs a delta per survivor and wins once sparsity is high.
The delta width is searched per layer, because a 4-bit delta spans 15 positions
and at 98% sparsity the average gap is far larger, so the encoder emits filler
entries that outnumber the real values three to one.

```bash
./venv/bin/python scripts/test_prune_encoding.py    # shows the crossover
```

### 6. `src/compress/weight_share.py` - stage 2

`kmeans_1d` is Lloyd's algorithm: assign every weight to its nearest centroid,
move each centroid to the mean of its members, repeat.

`entropy_constrained_kmeans` adds a rate term to the assignment step. Read the
docstring for why: plain k-means minimises distortion at a fixed number of
codes, but the pipeline pays for the **entropy** of the code stream, and
minimising inertia spreads weights evenly across clusters, which is the
distribution an entropy coder cannot compress.

```bash
./venv/bin/python scripts/test_entropy_interaction.py   # the measurement
./venv/bin/python scripts/test_ecsq.py                  # the rate-distortion curve
```

This is reported as a negative result: it loses to plain per-tensor linear
quantization on both axes, because it sacrifices the large-magnitude tail that
dominates the output.

### 7. `src/compress/huffman.py` - stage 3

`_huffman_lengths` builds the code by repeatedly merging the two rarest symbols.
`_canonical_codes` turns the resulting lengths into actual bit strings.

The one idea: **canonical** Huffman is reconstructible from the code lengths
alone, so the stored table is one small integer per symbol instead of a
serialised tree. That table is charged as storage.

Huffman can lose. On a short stream the table costs more than the saving, so the
pipeline only adopts it where it actually pays.

### 8. `src/compress/fold.py`

BatchNorm folded into the preceding convolution, exact at inference. Removes
three of the four stored vectors per BatchNorm layer.

### 9. `src/compress/sizing.py` - the accounting

Nothing is free. Value codes, position metadata, codebooks and scales, Huffman
tables, BatchNorm parameters **and** the `running_mean`/`running_var` buffers
that `model.parameters()` does not return.

The one idea: a term that is negligible before compression can dominate after
it. BatchNorm is 1.5% of the fp32 model and roughly 18% of the compressed one.

### 10. `src/compress/pipeline.py` - orchestration

Where the layer policy lives (which layers get which bit width, which are
exempt) and where the activation quantizers are attached.

The one mechanism worth understanding: activation quantizers attach through
**forward hooks**. A PyTorch forward hook that returns a value replaces the
module's output, so the quantizers splice into the network without a single line
of compression code in the model file.

`_encode_layer` searches layouts and entropy coding jointly, because they
interact: Huffman shrinks the value stream but not the bitmap, which moves the
crossover between layouts.

### 11. `src/compress/qat.py` - fine-tuning

One-shot compression at aggressive settings is unusable: at 70% sparsity and
3-bit weights the model sits at 10%, which is random guessing. Fine-tuning
recovers it to 93%.

The one idea: the **master-weight** straight-through estimator.

```
save w_master
w <- quantize(w)          run forward/backward here
w <- w_master             restore before the optimiser step
optimizer.step()          update the full-precision copy
```

Gradients are measured where the network actually operates, but accumulated in
full precision, so updates smaller than one quantization step are not lost.

Gradient clipping is not optional here. Without it the estimator diverges at low
bit width, and the failure is absorbing: a dead layer has no gradient, so no
later step can revive it.

```bash
./venv/bin/python scripts/diagnose_qat_collapse.py    # watch it happen
```

---

## Part 3: does any of it actually work

### 12. `scripts/conformance.py`

28 checks, architecture through accounting. Section G is the one that matters:
it takes the encoding the size figure claims, **decodes it**, and compares
against the weights the accuracy figure was measured from.

It exists because two bugs reached execution while printing entirely plausible
compression ratios. Both had the same root cause: encodings were being costed
but never decoded. See `LEARNINGS.md`.

```bash
./venv/bin/python scripts/conformance.py
```

---

## Suggested first exercise

```bash
./venv/bin/python scripts/compress_eval.py --weight-bits 4 --activation-bits 8 \
    --sparsity 0.8 --full-table
```

Then change one thing and predict the result before running it:

* `--weight-bits 3` - the ratio rises more than you expect. Why? (3-bit
  per-tensor quantization is itself a pruner: 88% of weights round to zero.)
* `--no-huffman` - which layers lose most? (The ones whose code distribution is
  most skewed.)
* `--sparsity 0.0` vs `0.3` - the ratio can get *worse*. Why? (Pruning removes
  the near-zero weights that were Huffman's cheapest symbol, and adds a bitmap.)

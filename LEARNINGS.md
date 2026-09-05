# LEARNINGS

Transferable rules discovered while building this assignment. Each one cost a
real bug or a wrong number; `scripts/conformance.py` enforces the checkable ones.

## L1. A compression figure is only real if a decoder reproduces the model

Two bugs shipped into a *running* pipeline that printed plausible compression
ratios, because the encoders were only ever **costed**, never **decoded**:

  * a `dense` encoding was offered for pruned layers, where the code sitting at
    a pruned position is meaningless (under k-means it is index 0, the
    most-negative centroid), so the claimed bitstream could not reconstruct the
    tensor;
  * relative-index "filler" entries carry a zero weight, but no k-means code
    mapped to exactly 0.0, so fillers would have decoded to a large negative
    value.

Neither produced an error, a warning, or an implausible number. The model
evaluated correctly in both cases - only the *claimed encoding* was wrong, and
the compression ratio is computed from the claim.

**Rule:** every encoder ships with a decoder, and the accounting is validated by
decoding the claimed bitstream and comparing against the weights the accuracy
number was produced from. Enforced by conformance section G.

## L2. A conformance check that greps text will fail on its own prose

The first forbidden-API scan flagged `prune.py` because its docstring says the
file does *not* use `torch.nn.utils.prune`. Regexes over source lines cannot
distinguish a call from a comment, a docstring, or a string literal.

**Rule:** scan the AST, not the text. Check `ast.Import` / `ast.ImportFrom`
nodes and attribute chains. Enforced by conformance section A, which also
asserts the scanner actually parsed a plausible number of files - a scanner that
silently parses nothing passes every check.

## L3. Measure the encoder parameters, never inherit them

Deep Compression specifies a 4-bit relative index. Carried over unexamined, that
choice wastes 60% of the encoding budget at 98% sparsity, because a 4-bit delta
spans 15 positions and the mean gap is ~50, so filler entries outnumber real
values 3:1. Searching the delta width per layer recovers 30% at 95% sparsity and
60% at 98%.

**Rule:** a constant taken from a paper is a hypothesis about *its* operating
point, not a setting for yours. Sweep it against your own data.

## L4. Metadata that is negligible before compression can dominate after it

BatchNorm holds 68,224 values including the `running_mean`/`running_var`
buffers, which `model.parameters()` does not report. At fp32 that is 1.5% of the
model - a rounding error. Against a compressed model of ~1 MB it is a
double-digit percentage.

The same effect hits k-means codebooks: a 256-entry fp32 codebook costs 8,192
bits, which *exceeds* the 6,912 bits of values in the 864-weight stem layer.
The stem compresses only 1.95x as a result.

**Rule:** account for every value a decoder needs, and re-check which terms
dominate *after* compression, not before. A per-layer method choice is required
whenever fixed metadata competes with a small payload.

## L5. Inertia is not accuracy

k-means with density initialisation reaches lower inertia (7.42) than linear
initialisation (7.99), but inertia weights every weight equally while the output
error is dominated by large-magnitude weights. Deep Compression's argument for
linear init is precisely this.

**Rule:** optimise the metric you are graded on. Decide initialisation by
measured top-1, not by the clustering objective.

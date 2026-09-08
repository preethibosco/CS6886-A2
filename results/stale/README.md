# Superseded results

These files were produced before the baseline bug was fixed.

`compress()` derived the fp32 baseline from whichever model it was handed.
Fine-tuning folds BatchNorm before `compress()` sees the model, so every run with
`fold_bn` was quoted against a baseline of 2,219,626 values instead of the
2,270,794 of the network that was actually trained. That understates the
compression ratio by 2.3% and makes the weight ratio collapse onto the model
ratio.

Only `fold_bn` runs were affected; runs that keep BatchNorm were measured
correctly. Kept for provenance, excluded from the report. Conformance section G3
now guards the baseline.

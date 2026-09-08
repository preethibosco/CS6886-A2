# Final results

Baseline MobileNet-v2 on CIFAR-10: **95.17%** top-1, 8.662 MB fp32 (parameters + BatchNorm buffers).

All rows are quantization-aware fine-tuned (30 epochs, gradient clipping 5.0).
`fold` = BatchNorm folded into the preceding convolution.
`*` marks the Pareto front (no other configuration is both smaller and more accurate).

| w | a | sparsity | fold | ratio | size (MB) | PTQ top-1 | QAT top-1 | vs fp32 | |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 8 | 0.90 | Y | 48.90x | 0.177 | 10.00 | **83.28** | -11.89 | * |
| 3 | 8 | 0.80 | - | 45.27x | 0.191 | 8.45 | **19.20** | -75.97 |  |
| 3 | 8 | 0.80 | Y | 38.67x | 0.224 | 10.12 | **92.77** | -2.40 | * |
| 3 | 8 | 0.70 | - | 37.29x | 0.232 | 31.07 | **91.99** | -3.18 |  |
| 3 | 4 | 0.70 | Y | 36.89x | 0.235 | 10.00 | **92.01** | -3.16 |  |
| 3 | 8 | 0.80 | - | 36.48x | 0.237 | 27.07 | **87.62** | -7.55 |  |
| 3 | 8 | 0.70 | Y | 35.90x | 0.241 | 10.00 | **93.21** | -1.96 | * |
| 3 | 8 | 0.50 | - | 33.07x | 0.262 | 40.43 | **91.09** | -4.08 |  |
| 3 | 8 | 0.50 | Y | 32.66x | 0.265 | 10.00 | **93.59** | -1.58 | * |
| 3 | 8 | 0.90 | - | 32.23x | 0.269 | 15.41 | **93.63** | -1.54 | * |
| 3 | 4 | 0.80 | - | 31.72x | 0.273 | 20.37 | **93.33** | -1.84 |  |
| 3 | 8 | 0.70 | - | 31.59x | 0.274 | 32.42 | **94.20** | -0.97 | * |
| 4 | 4 | 0.95 | - | 31.42x | 0.276 | 10.00 | **92.06** | -3.11 |  |
| 3 | 8 | 0.80 | - | 31.40x | 0.276 | 27.12 | **94.24** | -0.93 | * |
| 4 | 8 | 0.95 | - | 31.26x | 0.277 | 10.00 | **92.47** | -2.70 |  |
| 3 | 8 | 0.50 | - | 29.94x | 0.289 | 46.23 | **94.23** | -0.94 |  |
| 4 | 4 | 0.90 | - | 27.59x | 0.314 | 10.90 | **92.43** | -2.74 |  |
| 4 | 8 | 0.90 | - | 27.47x | 0.315 | 13.15 | **93.75** | -1.42 |  |
| 4 | 8 | 0.80 | Y | 25.58x | 0.339 | 10.01 | **94.09** | -1.08 |  |
| 4 | 8 | 0.80 | - | 23.31x | 0.372 | 45.18 | **94.73** | -0.44 | * |
| 3 | 8 | 0.80 | Y | 20.54x | 0.422 | 10.00 | **94.35** | -0.82 |  |
| 3 | 8 | 0.00 | - | 19.79x | 0.438 | 46.18 | **94.01** | -1.16 |  |
| 3 | 8 | 0.70 | Y | 17.60x | 0.492 | 41.62 | **94.68** | -0.49 |  |

## Recommended operating points

**within 1.0 point of fp32** - w3 / a8, sparsity 0.70

* model compression ratio: **31.59x**  (8.662 MB -> 0.274 MB)
* weight compression ratio: **40.48x**
* activation compression ratio: **4.00x** (32 / 8 bits, measured as total activation traffic per inference)
* top-1 after compression: **94.20%** (-0.97 vs fp32)
* same configuration without fine-tuning: 32.42%

**within 2.0 points of fp32** - w3 / a8, sparsity 0.70, BatchNorm folded

* model compression ratio: **35.90x**  (8.662 MB -> 0.241 MB)
* weight compression ratio: **35.09x**
* activation compression ratio: **4.00x** (32 / 8 bits, measured as total activation traffic per inference)
* top-1 after compression: **93.21%** (-1.96 vs fp32)
* same configuration without fine-tuning: 10.00%


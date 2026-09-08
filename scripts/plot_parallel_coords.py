"""Parallel-coordinates chart of the compression sweep (Assignment Q3b).

Built from results/sweep_results.json - the same 60 post-training-quantization
runs that were logged to Weights & Biases, so this figure and the wandb panel
show the same data. It is included because a static report needs a figure that
does not depend on a live wandb session.

Axes follow the assignment's sample plot: the knobs (activation bits, weight
bits) on the left, the outcomes (compression ratio, model size, accuracy) on the
right. Each line is one configuration.

Lines are coloured by accuracy on a single-hue sequential ramp (light -> dark =
worse -> better), because accuracy here is a continuous magnitude, not an
identity. Colour is redundant with the rightmost axis, so the chart is readable
without it.

Run:  python scripts/plot_parallel_coords.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
# Blue sequential ramp, steps 100 -> 700 of the reference palette.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)

blob = json.load(open("results/sweep_results.json"))
runs, baseline = blob["results"], blob["baseline"]["top1"]

AXES = [("activation_quant_bits", "activation\nbits", "{:.0f}"),
        ("weight_quant_bits", "weight\nbits", "{:.0f}"),
        ("compression_ratio", "compression\nratio", "{:.1f}x"),
        ("model_size_mb", "model size\n(MB)", "{:.2f}"),
        ("quantized_acc", "top-1 after\ncompression (%)", "{:.1f}")]

data = np.array([[r[k] for k, _, _ in AXES] for r in runs], dtype=float)
lo, hi = data.min(axis=0), data.max(axis=0)
span = np.where(hi - lo == 0, 1.0, hi - lo)
norm_data = (data - lo) / span

fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
x = np.arange(len(AXES))

acc = data[:, -1]
cnorm = Normalize(vmin=acc.min(), vmax=acc.max())
order = np.argsort(acc)          # draw good configurations last, on top
for i in order:
    ax.plot(x, norm_data[i], color=CMAP(cnorm(acc[i])), lw=1.0, alpha=0.75, zorder=2)

for j, (_, label, fmt) in enumerate(AXES):
    ax.axvline(j, color=GRID, lw=1.0, zorder=1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ax.text(j - 0.03, frac, fmt.format(lo[j] + frac * span[j]),
                ha="right", va="center", fontsize=7.5, color=INK_MUTED, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([label for _, label, _ in AXES], fontsize=9.5, color=INK)
ax.set_yticks([])
ax.set_xlim(-0.45, len(AXES) - 0.55)
ax.set_ylim(-0.06, 1.10)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0, colors=INK)

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=cnorm)
cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cb.set_label("top-1 after compression (%)", fontsize=8.5, color=INK_MUTED)
cb.ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=0)
cb.outline.set_visible(False)

ax.set_title(f"Compression sweep: {len(runs)} configurations, post-training quantization\n"
             f"fp32 baseline {baseline:.2f}% top-1",
             fontsize=11.5, color=INK, pad=14)
fig.tight_layout()
fig.savefig("results/figures/fig_parallel_coords.png", dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
fig.savefig("results/figures/fig_parallel_coords.pdf",
            bbox_inches="tight", facecolor=SURFACE)
print(f"wrote results/figures/fig_parallel_coords.png (+ .pdf)  [{len(runs)} configurations]")
print(f"  accuracy range {acc.min():.2f}% - {acc.max():.2f}%")
print(f"  ratio range    {data[:,2].min():.2f}x - {data[:,2].max():.2f}x")

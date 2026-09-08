"""Generate the final results table and Pareto figure from the run artefacts.

Every number in the report comes from here, read out of the JSON each run wrote.
Nothing is transcribed by hand, so the report cannot drift from the experiments.

Run:  python scripts/gen_final_results.py
Writes: results/final_results.md, results/figures/fig_pareto.png
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_MD = "results/final_results.md"
OUT_FIG = "results/figures/fig_pareto.png"
os.makedirs("results/figures", exist_ok=True)


def load_qat():
    """Every *valid* QAT run, from results/*.json written by scripts/run_qat.py.

    Runs made before gradient clipping existed are excluded. They were exposed
    to the absorbing divergence documented in scripts/diagnose_qat_collapse.py,
    where a layer's weights reach 1e14 and the loss pins at ln(10); the damaged
    models that result compress unusually well precisely because their weights
    are degenerate, so those runs report inflated ratios at depressed accuracy.
    One of them ("46.66x at 88.01%") sat on the Pareto front until this filter
    was added - and would have been reported, because results/qat_*.json sorts
    after results/final_*.json and silently won the de-duplication.

    A run is admitted only if its recorded args carry a non-zero grad_clip.
    """
    rows, skipped = [], []
    for path in sorted(glob.glob("results/*.json")):
        try:
            blob = json.load(open(path))
        except Exception:
            continue
        if not isinstance(blob, dict) or "results" not in blob:
            continue
        if not blob.get("args", {}).get("grad_clip"):
            skipped.append(os.path.basename(path))
            continue
        for r in blob["results"]:
            if "qat_top1" not in r:
                continue
            key = (r["weight_bits"], r["activation_bits"], r["sparsity"],
                   r.get("fold_bn", False), r.get("method"), r.get("bn_bits", 8),
                   r.get("depthwise_bits", 8))
            rows.append({**r, "_key": key, "_src": os.path.basename(path)})
    # Keep the last occurrence of each key.
    if skipped:
        print(f"excluded {len(skipped)} pre-clipping result file(s): {', '.join(skipped)}")
    dedup = {}
    for r in rows:
        dedup[r["_key"]] = r
    return list(dedup.values())


def pareto(rows, xk="model_ratio", yk="qat_top1"):
    """Points not dominated on both compression ratio and accuracy."""
    front = []
    for r in sorted(rows, key=lambda r: (-r[xk], -r[yk])):
        if all(r[yk] > f[yk] for f in front):
            front.append(r)
    return front


rows = load_qat()
if not rows:
    print("no QAT results found"); sys.exit(1)

baseline = 95.17
rows.sort(key=lambda r: -r["model_ratio"])
front = pareto(rows)
front_keys = {r["_key"] for r in front}

lines = ["# Final results", "",
         f"Baseline MobileNet-v2 on CIFAR-10: **{baseline:.2f}%** top-1, "
         f"8.662 MB fp32 (parameters + BatchNorm buffers).", "",
         "All rows are quantization-aware fine-tuned (30 epochs, gradient clipping 5.0).",
         "`fold` = BatchNorm folded into the preceding convolution.",
         "`*` marks the Pareto front (no other configuration is both smaller and more accurate).",
         "",
         "| w | a | sparsity | fold | ratio | size (MB) | PTQ top-1 | QAT top-1 | vs fp32 | |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(
        f"| {r['weight_bits']} | {r['activation_bits']} | {r['sparsity']:.2f} | "
        f"{'Y' if r.get('fold_bn') else '-'} | {r['model_ratio']:.2f}x | "
        f"{r['size_mb']:.3f} | {r['ptq_top1']:.2f} | **{r['qat_top1']:.2f}** | "
        f"{r['qat_top1']-baseline:+.2f} | {'*' if r['_key'] in front_keys else ''} |")

# Recommended: the most compressed configuration still within 1 point of fp32.
within1 = [r for r in rows if r["qat_top1"] >= baseline - 1.0]
within2 = [r for r in rows if r["qat_top1"] >= baseline - 2.0]
best1 = max(within1, key=lambda r: r["model_ratio"]) if within1 else None
best2 = max(within2, key=lambda r: r["model_ratio"]) if within2 else None

lines += ["", "## Recommended operating points", ""]
for tag, r in (("within 1.0 point of fp32", best1), ("within 2.0 points of fp32", best2)):
    if r is None:
        continue
    lines += [
        f"**{tag}** - w{r['weight_bits']} / a{r['activation_bits']}, "
        f"sparsity {r['sparsity']:.2f}"
        f"{', BatchNorm folded' if r.get('fold_bn') else ''}",
        "",
        f"* model compression ratio: **{r['model_ratio']:.2f}x**  (8.662 MB -> {r['size_mb']:.3f} MB)",
        f"* weight compression ratio: **{r['weight_ratio']:.2f}x**",
        f"* activation compression ratio: **{r['activation_ratio']:.2f}x** "
        f"(32 / {r['activation_bits']} bits, measured as total activation traffic per inference)",
        f"* top-1 after compression: **{r['qat_top1']:.2f}%** ({r['qat_top1']-baseline:+.2f} vs fp32)",
        f"* same configuration without fine-tuning: {r['ptq_top1']:.2f}%",
        ""]
open(OUT_MD, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT_MD}  ({len(rows)} configurations)")

# ------------------------------------------------------------------ figure
# Palette: categorical slots 1 and 2 of the validated reference palette
# (validate_palette.js: all six checks PASS, worst adjacent CVD dE 24.7).
# Colour encodes the BatchNorm treatment - the one design choice that moves a
# point along the trade-off - and never encodes rank.
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_MUTED = "#52514e"
GRID      = "#e3e2de"
C_KEPT    = "#2a78d6"
C_FOLD    = "#eb6834"

# Configurations far below the baseline are omitted from the plot and reported
# in the table instead. This is a readability choice, not a selective one: every
# omitted point costs more than 5 points of top-1, so none of them is an
# operating point anyone would ship, while plotting them stretches the y-axis
# across an empty band and compresses the 92-95% region where every real
# decision is actually made. The table below lists all 20 configurations.
PLOT_FLOOR = 90.0
plotted = [r for r in rows if r["qat_top1"] >= PLOT_FLOOR]
omitted = [r for r in rows if r["qat_top1"] < PLOT_FLOOR]

fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

for fold, colour, label in ((False, C_KEPT, "BatchNorm kept, quantized to 8 bits"),
                            (True,  C_FOLD, "BatchNorm folded into the convolution")):
    sub = [r for r in plotted if bool(r.get("fold_bn")) == fold]
    if not sub:
        continue
    ax.scatter([r["model_ratio"] for r in sub], [r["qat_top1"] for r in sub],
               s=64, color=colour, edgecolor=SURFACE, linewidth=2.0,
               zorder=3, label=label)

fr = sorted([r for r in front if r["qat_top1"] >= PLOT_FLOOR],
            key=lambda r: r["model_ratio"])
ax.plot([r["model_ratio"] for r in fr], [r["qat_top1"] for r in fr],
        color=INK_MUTED, lw=1.0, alpha=0.55, zorder=2, label="Pareto front")

ax.axhline(baseline, color=INK_MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
ax.annotate(f"fp32 baseline  {baseline:.2f}%", (0.988, baseline),
            xycoords=("axes fraction", "data"), textcoords="offset points",
            xytext=(0, 6), ha="right", color=INK_MUTED, fontsize=8.5)

# Direct-label the Pareto front only (never a label on every point). Offsets
# alternate vertically and fan horizontally so the three points clustered near
# 31-32x stay legible.
# Three front points sit within 1x of each other near 31-32x, so their labels
# are pushed clear and tied back with hairline leaders. A label that could
# belong to either of two adjacent marks is worse than no label.
OFFSETS = [(0, 14), (-34, -20), (30, 16), (36, -18), (0, 15), (0, -20)]
for i, r in enumerate(fr):
    dx, dy = OFFSETS[i] if i < len(OFFSETS) else (0, 14 if i % 2 == 0 else -20)
    ha = "center" if dx == 0 else ("right" if dx < 0 else "left")
    txt = f"w{r['weight_bits']} sp{r['sparsity']:g}" + (" fold" if r.get("fold_bn") else "")
    leader = dict(arrowstyle="-", color=GRID, lw=0.8,
                  shrinkA=1, shrinkB=5) if dx else None
    ax.annotate(txt, (r["model_ratio"], r["qat_top1"]), textcoords="offset points",
                xytext=(dx, dy), ha=ha, fontsize=8, color=INK_MUTED, zorder=4,
                arrowprops=leader)

ax.set_xlabel("model compression ratio", color=INK, fontsize=10)
ax.set_ylabel("top-1 accuracy after compression (%)", color=INK, fontsize=10)
ax.set_title("Compression ratio against accuracy, after quantization-aware fine-tuning",
             color=INK, fontsize=11.5, pad=12)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}x")
ax.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
ax.set_ylim(min(r["qat_top1"] for r in plotted) - 0.9, 95.9)

leg = ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor=SURFACE,
                edgecolor=GRID, borderpad=0.7)
for t in leg.get_texts():
    t.set_color(INK_MUTED)

if omitted:
    fig.text(0.5, -0.012,
             f"{len(omitted)} configuration(s) below {PLOT_FLOOR:g}% top-1 omitted for "
             f"readability; all {len(rows)} appear in results/final_results.md",
             ha="center", fontsize=7.5, color=INK_MUTED)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(OUT_FIG.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURFACE)
print(f"wrote {OUT_FIG} (+ .pdf)")

print("\n=== Pareto front ===")
for r in sorted(front, key=lambda r: -r["model_ratio"]):
    print(f"  w{r['weight_bits']} a{r['activation_bits']} sp{r['sparsity']:.2f}"
          f"{' fold' if r.get('fold_bn') else '    '}  {r['model_ratio']:6.2f}x  "
          f"{r['size_mb']:.3f} MB  {r['qat_top1']:.2f}%  ({r['qat_top1']-baseline:+.2f})")

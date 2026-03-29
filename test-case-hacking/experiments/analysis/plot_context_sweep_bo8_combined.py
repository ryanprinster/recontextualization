#!/usr/bin/env python3
"""Plot thinking vs no-think BoN=8 context sweep results on the same axes."""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/workspace/experiments")

# -----------------------------
# GLOBAL STYLE
# -----------------------------
sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.linewidth": 1,
    "axes.edgecolor": "black",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})

# -----------------------------
# COLOR PALETTES
# -----------------------------
greens = sns.color_palette("Greens", 6)
purples = sns.color_palette("Purples", 6)

COLORS = {
    "N\u2192N": purples[2],
    "NH\u2192NH": purples[4],
    "H\u2192H": purples[5],
    "N\u2192H": greens[2],
    "NH\u2192N": greens[4],
    "NH\u2192H": greens[5],
}

MARKERS = {
    "think": "D",      # filled diamond
    "nothink": "d",    # thin diamond
}

GROUPS = ["N_to_N", "NH_to_NH", "H_to_H", "N_to_H", "NH_to_N", "NH_to_H"]
GROUP_LABELS = {
    "N_to_N": "N\u2192N", "NH_to_NH": "NH\u2192NH", "H_to_H": "H\u2192H",
    "N_to_H": "N\u2192H", "NH_to_N": "NH\u2192N", "NH_to_H": "NH\u2192H",
}


# -----------------------------
# PLOT FUNCTION
# -----------------------------
def plot_point(ax, x, y, xerr, yerr, color, marker, label=None, size=120):
    ax.errorbar(
        x, y,
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor="gray",
        elinewidth=1.4,
        capsize=3,
        alpha=0.9,
        zorder=1,
    )
    ax.scatter(
        x, y,
        s=size,
        marker=marker,
        color=color,
        edgecolor="black",
        linewidth=1.1,
        alpha=0.85,
        zorder=3,
        label=label,
    )


def binomial_ci_95(count, n):
    p = count / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lo = max(0, centre - margin) * 100
    hi = min(1, centre + margin) * 100
    pct = p * 100
    return np.array([[pct - lo], [hi - pct]])


def load_experiment(base_dir):
    data = {}
    for group in GROUPS:
        report_path = glob.glob(f"{base_dir}/{group}/post_training_evaluation/standard/evaluation_report_*.json")[0]
        with open(report_path) as f:
            m = json.load(f)["context_metrics"]["standard"]
        data[group] = {
            "n": m["total_samples"],
            "correct_pct": m["category_percentages"]["correct"],
            "hack_pct": m["category_percentages"].get("hacking", 0.0),
            "correct_count": m["category_counts"]["correct"],
            "hack_count": m["category_counts"].get("hacking", 0),
        }
    # Base model
    base_path = glob.glob(f"{base_dir}/NH_to_H/evaluation/standard/evaluation_report_*.json")[0]
    with open(base_path) as f:
        bm = json.load(f)["context_metrics"]["standard"]
    base = {
        "n": bm["total_samples"],
        "correct_pct": bm["category_percentages"]["correct"],
        "hack_pct": bm["category_percentages"].get("hacking", 0.0),
        "correct_count": bm["category_counts"]["correct"],
        "hack_count": bm["category_counts"].get("hacking", 0),
    }
    return data, base


# --- Load both experiments ---
think_data, think_base = load_experiment(f"{OUTPUT_ROOT}/training/context_sweep_v2_think_bo8")
nothink_data, nothink_base = load_experiment(f"{OUTPUT_ROOT}/training/context_sweep_v2_nothink_bo8")

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 7))

# Plot thinking points
for group in GROUPS:
    label = GROUP_LABELS[group]
    d = think_data[group]
    plot_point(
        ax,
        d["hack_pct"], d["correct_pct"],
        binomial_ci_95(d["hack_count"], d["n"]),
        binomial_ci_95(d["correct_count"], d["n"]),
        COLORS[label], MARKERS["think"],
    )
    ax.annotate(
        label, (d["hack_pct"], d["correct_pct"]),
        textcoords="offset points", xytext=(8, -12),
        fontsize=8.5, fontweight="bold", color=COLORS[label],
    )

# Plot no-think points
for group in GROUPS:
    label = GROUP_LABELS[group]
    d = nothink_data[group]
    plot_point(
        ax,
        d["hack_pct"], d["correct_pct"],
        binomial_ci_95(d["hack_count"], d["n"]),
        binomial_ci_95(d["correct_count"], d["n"]),
        COLORS[label], MARKERS["nothink"],
    )
    ax.annotate(
        label, (d["hack_pct"], d["correct_pct"]),
        textcoords="offset points", xytext=(8, 6),
        fontsize=8.5, fontweight="bold", color=COLORS[label],
    )

# Draw lines connecting think -> nothink for each group
for group in GROUPS:
    label = GROUP_LABELS[group]
    td = think_data[group]
    nd = nothink_data[group]
    ax.plot(
        [td["hack_pct"], nd["hack_pct"]],
        [td["correct_pct"], nd["correct_pct"]],
        color=COLORS[label], linestyle="--", linewidth=1.0, alpha=0.5, zorder=0,
    )

# Base model points
ax.scatter(
    think_base["hack_pct"], think_base["correct_pct"],
    marker="X", s=220, color="dimgray", edgecolor="black",
    linewidth=1.2, zorder=4,
)
ax.annotate(
    "Base (think)", (think_base["hack_pct"], think_base["correct_pct"]),
    textcoords="offset points", xytext=(8, -12),
    fontsize=8.5, fontweight="bold", color="dimgray",
)

ax.scatter(
    nothink_base["hack_pct"], nothink_base["correct_pct"],
    marker="X", s=220, color="silver", edgecolor="black",
    linewidth=1.2, zorder=4,
)
ax.annotate(
    "Base (no-think)", (nothink_base["hack_pct"], nothink_base["correct_pct"]),
    textcoords="offset points", xytext=(8, 6),
    fontsize=8.5, fontweight="bold", color="gray",
)

# --- Legend ---
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([], [], marker="X", color="gray", markeredgecolor="black",
           markersize=10, linestyle="none", label="Base model"),
    Line2D([], [], marker=MARKERS["think"], color="gray", markeredgecolor="black",
           markersize=10, linestyle="none", label="Thinking (BoN=8)"),
    Line2D([], [], marker=MARKERS["nothink"], color="gray", markeredgecolor="black",
           markersize=10, linestyle="none", label="No-think (BoN=8)"),
    Line2D([], [], marker="none", linestyle="none", label=""),
    Line2D([], [], marker="s", color=purples[3], markeredgecolor="black",
           markersize=8, linestyle="none", label="Standard"),
    Line2D([], [], marker="s", color=greens[4], markeredgecolor="black",
           markersize=8, linestyle="none", label="Recontextualized"),
]

ax.legend(handles=legend_elements, loc="lower left", framealpha=0.9, fontsize=10)

ax.set_xlabel("Hack (%)")
ax.set_ylabel("Correct (%)")
ax.set_title("Context Sweep BoN=8: Thinking vs No-Think")

plt.tight_layout()
plt.savefig(f"{OUTPUT_ROOT}/context_sweep_bo8_combined_results.png", dpi=150)
print(f"Saved to {OUTPUT_ROOT}/context_sweep_bo8_combined_results.png")

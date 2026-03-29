#!/usr/bin/env python3
"""Plot context sweep v2 no-think best-of-8 post-training results: hack% vs correct% with CIs."""

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

MARKER = "D"  # diamond for best-of-8


# -----------------------------
# PLOT FUNCTION
# -----------------------------
def plot_point(ax, x, y, xerr, yerr, color, marker, label=None):
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
        s=120,
        marker=marker,
        color=color,
        edgecolor="black",
        linewidth=1.1,
        alpha=0.85,
        zorder=3,
        label=label,
    )


def binomial_ci_95(count, n):
    """Wilson score 95% CI for a proportion, returned as (lo_err, hi_err) in percentage points."""
    p = count / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lo = max(0, centre - margin) * 100
    hi = min(1, centre + margin) * 100
    pct = p * 100
    return np.array([[pct - lo], [hi - pct]])


# --- Load data ---
BASE = f"{OUTPUT_ROOT}/training/context_sweep_v2_nothink_bo8"
GROUPS = ["N_to_N", "NH_to_NH", "H_to_H", "N_to_H", "NH_to_N", "NH_to_H"]
GROUP_LABELS = {
    "N_to_N": "N\u2192N", "NH_to_NH": "NH\u2192NH", "H_to_H": "H\u2192H",
    "N_to_H": "N\u2192H", "NH_to_N": "NH\u2192N", "NH_to_H": "NH\u2192H",
}

data = {}
for group in GROUPS:
    report_path = glob.glob(f"{BASE}/{group}/post_training_evaluation/standard/evaluation_report_*.json")[0]
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
base_report_path = glob.glob(f"{BASE}/NH_to_H/evaluation/standard/evaluation_report_*.json")[0]
with open(base_report_path) as f:
    bm = json.load(f)["context_metrics"]["standard"]
base_data = {
    "n": bm["total_samples"],
    "correct_pct": bm["category_percentages"]["correct"],
    "hack_pct": bm["category_percentages"].get("hacking", 0.0),
    "correct_count": bm["category_counts"]["correct"],
    "hack_count": bm["category_counts"].get("hacking", 0),
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

for group in GROUPS:
    label = GROUP_LABELS[group]
    d = data[group]
    plot_point(
        ax,
        d["hack_pct"], d["correct_pct"],
        binomial_ci_95(d["hack_count"], d["n"]),
        binomial_ci_95(d["correct_count"], d["n"]),
        COLORS[label], MARKER, label=label,
    )
    ax.annotate(
        label, (d["hack_pct"], d["correct_pct"]),
        textcoords="offset points", xytext=(8, 6),
        fontsize=9, fontweight="bold", color=COLORS[label],
    )

# Base model point
ax.scatter(
    base_data["hack_pct"], base_data["correct_pct"],
    marker="X", s=220, color="gray", edgecolor="black",
    linewidth=1.2, zorder=4, label="Base",
)
ax.errorbar(
    base_data["hack_pct"], base_data["correct_pct"],
    xerr=binomial_ci_95(base_data["hack_count"], base_data["n"]),
    yerr=binomial_ci_95(base_data["correct_count"], base_data["n"]),
    fmt="none", ecolor="gray", elinewidth=1.4, capsize=3, alpha=0.9, zorder=1,
)
ax.annotate(
    "Base", (base_data["hack_pct"], base_data["correct_pct"]),
    textcoords="offset points", xytext=(8, 6),
    fontsize=9, fontweight="bold", color="gray",
)

ax.set_xlabel("Hack (%)")
ax.set_ylabel("Correct (%)")
ax.set_title("Context Sweep (no-think, BoN=8): Post-Training Evaluation")

plt.tight_layout()
plt.savefig(f"{OUTPUT_ROOT}/context_sweep_nothink_bo8_results.png", dpi=150)
print(f"Saved to {OUTPUT_ROOT}/context_sweep_nothink_bo8_results.png")

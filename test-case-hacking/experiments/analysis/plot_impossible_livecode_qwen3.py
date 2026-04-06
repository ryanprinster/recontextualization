#!/usr/bin/env python3
"""Plot Impossible LiveCode evaluation: hack% vs correct% with CIs across Qwen3 model sizes."""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.join(os.path.dirname(OUTPUT_ROOT), "evaluation")

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
# HELPERS
# -----------------------------
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


# -----------------------------
# DATA CONFIGS
# -----------------------------
purples = sns.color_palette("Purples", 8)

# Model sizes ordered largest to smallest, with darker colors for larger models
MODELS = [
    ("8B",   "o", 7),
    ("4B",   "s", 6),
    ("1.7B", "^", 5),
    ("0.6B", "v", 4),
]

# Map size labels to eval directory names
DIR_NAMES = {
    "8B": "qwen3_8b",
    "4B": "qwen3_4b",
    "1.7B": "qwen3_1_7b",
    "0.6B": "qwen3_0_6b",
}

configs = {}
for size, marker, shade in MODELS:
    dname = DIR_NAMES[size]
    configs[f"Qwen3-{size}"] = {
        "glob": f"{EVAL_ROOT}/impossible_livecode_{dname}/*/evaluation/standard/evaluation_report_*.json",
        "color": purples[shade],
        "marker": marker,
    }

# -----------------------------
# LOAD DATA
# -----------------------------
data = {}
for label, cfg in configs.items():
    reports = sorted(glob.glob(cfg["glob"]))
    if not reports:
        print(f"WARNING: No reports found for {label}: {cfg['glob']}")
        continue
    report_path = reports[-1]
    print(f"{label}: {report_path}")
    with open(report_path) as f:
        m = json.load(f)["context_metrics"]["standard"]
    data[label] = {
        "n": m["total_samples"],
        "correct_pct": m["category_percentages"]["correct"],
        "hack_pct": m["category_percentages"].get("hacking", 0.0),
        "correct_count": m["category_counts"]["correct"],
        "hack_count": m["category_counts"].get("hacking", 0),
        "color": cfg["color"],
        "marker": cfg["marker"],
    }

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 7))

for label, d in data.items():
    plot_point(
        ax,
        d["hack_pct"], d["correct_pct"],
        binomial_ci_95(d["hack_count"], d["n"]),
        binomial_ci_95(d["correct_count"], d["n"]),
        d["color"], d["marker"], label=label, size=160,
    )
    ax.annotate(
        label, (d["hack_pct"], d["correct_pct"]),
        textcoords="offset points", xytext=(10, 8),
        fontsize=10, fontweight="bold", color=d["color"],
    )

ax.set_xlabel("Hack (%)")
ax.set_ylabel("Correct (%)")
ax.set_title("Impossible LiveCode: Qwen3 Baseline (Thinking, standard context)")
ax.legend(loc="best", fontsize=10)

plt.tight_layout()
out_path = os.path.join(OUTPUT_ROOT, "impossible_livecode_qwen3.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")

#!/usr/bin/env python3
"""Plot context sweep post-training results: hack% vs correct% with CIs."""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/workspace/experiments")

# --- Load data ---
BASE = f"{OUTPUT_ROOT}/context_sweep"
GROUPS = ["N_to_N", "NH_to_NH", "H_to_H", "N_to_H", "NH_to_N", "NH_to_H"]

data = {}
for group in GROUPS:
    report_path = glob.glob(f"{BASE}/{group}/post_training_evaluation/standard/evaluation_report_*.json")[0]
    with open(report_path) as f:
        m = json.load(f)["context_metrics"]["standard"]
    n = m["total_samples"]
    correct_pct = m["category_percentages"]["correct"]
    hack_pct = m["category_percentages"]["hacking"]
    data[group] = {
        "n": n,
        "correct_pct": correct_pct,
        "hack_pct": hack_pct,
        "correct_count": m["category_counts"]["correct"],
        "hack_count": m["category_counts"]["hacking"],
    }

# Load base model (no training) evaluation
base_report_path = glob.glob(f"{BASE}/NH_to_H/evaluation/standard/evaluation_report_*.json")[0]
with open(base_report_path) as f:
    bm = json.load(f)["context_metrics"]["standard"]
base_data = {
    "n": bm["total_samples"],
    "correct_pct": bm["category_percentages"]["correct"],
    "hack_pct": bm["category_percentages"]["hacking"],
    "correct_count": bm["category_counts"]["correct"],
    "hack_count": bm["category_counts"]["hacking"],
}


def binomial_ci_95(count, n):
    """Wilson score 95% CI for a proportion, returned as percentage half-width."""
    p = count / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lo = max(0, centre - margin) * 100
    hi = min(1, centre + margin) * 100
    pct = p * 100
    return pct - lo, hi - pct


# --- Style ---
sns.set_theme(style="whitegrid", font_scale=1.1)

# Group definitions: (group_key, label, color, category_label)
plot_groups = [
    # Standard (purples)
    ("N_to_N",   "N->N",     "#c4b0d9", "Standard"),   # light purple
    ("NH_to_NH", "NH->NH",   "#7e57c2", "Standard"),   # medium purple
    ("H_to_H",   "H->H",     "#4a148c", "Standard"),   # dark purple
    # Recontextualized (greens)
    ("NH_to_N",  "NH->N",    "#81c784", "Recontextualized"),  # normalish green
    ("N_to_H",   "N->H",     "#a5d6a7", "Recontextualized"),  # light green
    ("NH_to_H",  "NH->H",    "#1b5e20", "Recontextualized"),  # dark green
]

fig, ax = plt.subplots(figsize=(7, 5.5))

# Track legend handles per category
legend_handles = {}

for group_key, label, color, cat in plot_groups:
    d = data[group_key]
    x = d["hack_pct"]
    y = d["correct_pct"]
    xerr = np.array([[v] for v in binomial_ci_95(d["hack_count"], d["n"])]).reshape(2, 1)
    yerr = np.array([[v] for v in binomial_ci_95(d["correct_count"], d["n"])]).reshape(2, 1)

    h = ax.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt="^",  # triangle marker
        color=color,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=0.6,
        capsize=3,
        capthick=1.2,
        elinewidth=1.2,
        label=label,
    )

    # Annotate each point
    ax.annotate(
        label, (x, y),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )

    # Collect for grouped legend
    if cat not in legend_handles:
        legend_handles[cat] = []
    legend_handles[cat].append(h)

# --- Plot base model point ---
bx = base_data["hack_pct"]
by = base_data["correct_pct"]
bxerr = np.array([[v] for v in binomial_ci_95(base_data["hack_count"], base_data["n"])]).reshape(2, 1)
byerr = np.array([[v] for v in binomial_ci_95(base_data["correct_count"], base_data["n"])]).reshape(2, 1)
base_handle = ax.errorbar(
    bx, by,
    xerr=bxerr, yerr=byerr,
    fmt="o",
    color="#888888",
    markersize=10,
    markeredgecolor="white",
    markeredgewidth=0.6,
    capsize=3,
    capthick=1.2,
    elinewidth=1.2,
    label="Base",
)
ax.annotate(
    "Base", (bx, by),
    textcoords="offset points",
    xytext=(8, 6),
    fontsize=8.5,
    color="#888888",
    fontweight="bold",
)

# --- Build grouped legend ---
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

legend_elements = []
legend_labels = []

# Base model entry
legend_elements.append(base_handle)
legend_labels.append("Base")

for cat in ["Standard", "Recontextualized"]:
    # Category header as a blank entry
    legend_elements.append(Line2D([], [], marker="none", linestyle="none"))
    legend_labels.append(cat)
    for h in legend_handles[cat]:
        legend_elements.append(h)
        legend_labels.append(f"  {h.get_label()}")

ax.legend(legend_elements, legend_labels, loc="best", framealpha=0.9, fontsize=9)

ax.set_xlabel("Hack %")
ax.set_ylabel("Correct %")
ax.set_title("Context Sweep: Post-Training Evaluation (standard context)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_ROOT}/context_sweep_results.png", dpi=150)
print(f"Saved to {OUTPUT_ROOT}/context_sweep_results.png")

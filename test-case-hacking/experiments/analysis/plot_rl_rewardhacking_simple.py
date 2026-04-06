#!/usr/bin/env python3
"""Plot rl_rewardhacking Simple: hack% vs correct% with CIs across Qwen3 model sizes."""

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
    """Wilson score 95% CI, returned as (lo_err, hi_err) in percentage points."""
    if n == 0:
        return np.array([[0], [0]])
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
        x, y, xerr=xerr, yerr=yerr,
        fmt="none", ecolor="gray", elinewidth=1.4, capsize=3, alpha=0.9, zorder=1,
    )
    ax.scatter(
        x, y, s=size, marker=marker, color=color,
        edgecolor="black", linewidth=1.1, alpha=0.85, zorder=3, label=label,
    )


def load_report(glob_pattern):
    """Load the latest evaluation report matching the pattern."""
    reports = sorted(glob.glob(glob_pattern))
    if not reports:
        return None
    with open(reports[-1]) as f:
        return json.load(f)["context_metrics"]["standard"]


# -----------------------------
# MODEL DEFINITIONS
# -----------------------------
MODELS = [
    ("32B",  "D", 7, "qwen3_32b"),
    ("8B",   "o", 6, "qwen3_8b"),
    ("4B",   "s", 5, "qwen3_4b"),
    ("1.7B", "^", 4, "qwen3_1_7b"),
    ("0.6B", "v", 3, "qwen3_0_6b"),
]

purples = sns.color_palette("Purples", 8)
greens = sns.color_palette("Greens", 8)

# -----------------------------
# BUILD CONFIGS
# -----------------------------
configs = {}
for size_label, marker, shade, dname in MODELS:
    configs[f"Qwen3-{size_label} Thinking"] = {
        "glob": f"{EVAL_ROOT}/rl_rewardhacking_simple_{dname}_think/*/evaluation/standard/evaluation_report_*.json",
        "color": purples[shade],
        "marker": marker,
    }
    configs[f"Qwen3-{size_label} Non-thinking"] = {
        "glob": f"{EVAL_ROOT}/rl_rewardhacking_simple_{dname}_nothink/*/evaluation/standard/evaluation_report_*.json",
        "color": greens[shade],
        "marker": marker,
    }

# -----------------------------
# LOAD DATA
# -----------------------------
data = {}
for label, cfg in configs.items():
    m = load_report(cfg["glob"])
    if m is None:
        print(f"WARNING: No reports for {label}")
        continue
    print(f"{label}: n={m['total_samples']}, correct={m['category_percentages']['correct']:.1f}%, hack={m['category_percentages'].get('hacking', 0):.1f}%")
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
        fontsize=9, fontweight="bold", color=d["color"],
    )

ax.set_xlabel("Hack (%)")
ax.set_ylabel("Correct (%)")
ax.set_title("RL Reward Hacking — SimpleOverwriteTests: Qwen3 (standard context)")
ax.legend(loc="best", fontsize=8, ncol=2)

plt.tight_layout()
out_path = os.path.join(OUTPUT_ROOT, "rl_rewardhacking_simple_qwen3.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")

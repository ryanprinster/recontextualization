#!/usr/bin/env python3
"""Plot rl-rewardhacking baseline: Qwen3 + gpt-oss-20b, split by hint type (simple vs aware)."""

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


def plot_point(ax, x, y, xerr, yerr, color, marker, label=None, size=120, alpha=0.85):
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt="none", ecolor="gray", elinewidth=1.4, capsize=3, alpha=0.9, zorder=1,
    )
    ax.scatter(
        x, y, s=size, marker=marker, color=color,
        edgecolor="black", linewidth=1.1, alpha=alpha, zorder=3, label=label,
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
QWEN3_MODELS = [
    ("32B",  "D", 7, "qwen3_32b"),
    ("8B",   "o", 6, "qwen3_8b"),
    ("4B",   "s", 5, "qwen3_4b"),
    ("1.7B", "^", 4, "qwen3_1_7b"),
    ("0.6B", "v", 3, "qwen3_0_6b"),
]

GPT_OSS_LEVELS = [
    ("Low",    "v", 3, "low"),
    ("Medium", "s", 5, "medium"),
    ("High",   "D", 7, "high"),
]

HINT_TYPES = {
    "simple": "SimpleOverwriteTests",
    "aware": "SimpleOverwriteTestsAware",
}

# Color palettes — simple uses filled markers, aware uses open/lighter markers
purples = sns.color_palette("Purples", 8)
greens = sns.color_palette("Greens", 8)
blues = sns.color_palette("Blues", 8)
# Aware variants: oranges/reds
oranges = sns.color_palette("Oranges", 8)
reds = sns.color_palette("Reds", 8)
teals = sns.color_palette("YlGnBu", 8)

# -----------------------------
# BUILD CONFIGS
# -----------------------------
configs = {}

for size_label, marker, shade, dname in QWEN3_MODELS:
    for hint_key, hint_label in [("simple", "Simple"), ("aware", "Aware")]:
        for mode, mode_label, color_pal in [("think", "Think", purples if hint_key == "simple" else oranges),
                                             ("nothink", "NoThink", greens if hint_key == "simple" else reds)]:
            suffix = "" if mode == "think" else "_nothink"
            label = f"Qwen3-{size_label} {mode_label} [{hint_label}]"
            configs[label] = {
                "glob": f"{EVAL_ROOT}/rl_rewardhacking_{hint_key}_{dname}_{mode}/*/evaluation/standard/evaluation_report_*.json",
                "color": color_pal[shade],
                "marker": marker,
                "hint_type": hint_key,
            }

for level_label, marker, shade, level_key in GPT_OSS_LEVELS:
    for hint_key, hint_label in [("simple", "Simple"), ("aware", "Aware")]:
        color_pal = blues if hint_key == "simple" else teals
        label = f"gpt-oss-20b ({level_label}) [{hint_label}]"
        configs[label] = {
            "glob": f"{EVAL_ROOT}/rl_rewardhacking_{hint_key}_gpt_oss_20b_think_{level_key}/*/evaluation/standard/evaluation_report_*.json",
            "color": color_pal[shade],
            "marker": marker,
            "hint_type": hint_key,
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
    print(f"{label}: n={m['total_samples']}")
    data[label] = {
        "n": m["total_samples"],
        "correct_pct": m["category_percentages"]["correct"],
        "hack_pct": m["category_percentages"].get("hacking", 0.0),
        "correct_count": m["category_counts"]["correct"],
        "hack_count": m["category_counts"].get("hacking", 0),
        "color": cfg["color"],
        "marker": cfg["marker"],
        "hint_type": cfg["hint_type"],
    }

if not data:
    print("ERROR: No evaluation data found. Run baselines first.")
    exit(1)

# -----------------------------
# PLOT — side by side subplots
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

for ax, hint_key, title in [
    (axes[0], "simple", "SimpleOverwriteTests (hint only)"),
    (axes[1], "aware", "SimpleOverwriteTestsAware (hint + warning)"),
]:
    subset = {k: v for k, v in data.items() if v["hint_type"] == hint_key}
    for label, d in subset.items():
        short_label = label.replace(" [Simple]", "").replace(" [Aware]", "")
        plot_point(
            ax, d["hack_pct"], d["correct_pct"],
            binomial_ci_95(d["hack_count"], d["n"]),
            binomial_ci_95(d["correct_count"], d["n"]),
            d["color"], d["marker"], label=short_label, size=160,
        )
        ax.annotate(
            short_label, (d["hack_pct"], d["correct_pct"]),
            textcoords="offset points", xytext=(10, 8),
            fontsize=8, fontweight="bold", color=d["color"],
        )
    ax.set_xlabel("Hack (%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7, ncol=2)

axes[0].set_ylabel("Correct (%)")
fig.suptitle("RL Reward Hacking Baseline: Hack% vs Correct% (standard context)", fontsize=15, y=1.02)

plt.tight_layout()
out_path = os.path.join(OUTPUT_ROOT, "rl_rewardhacking_baseline_overlay.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")

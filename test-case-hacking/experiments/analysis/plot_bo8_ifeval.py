#!/usr/bin/env python3
"""Plot IFEval results for BO8 training experiment: 6 arms as grouped bar chart."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

IFEVAL_BASE = Path(f"{OUTPUT_ROOT}/training/bo8_ifeval/ifeval_results")

# The 6 arms in display order
ARMS = [
    ("base_think",      "Base\n(think)"),
    ("base_nothink",    "Base\n(nothink)"),
    ("think_N_to_N",    "Think\nN\u2192N"),
    ("think_NH_to_H",   "Think\nNH\u2192H"),
    ("nothink_N_to_N",  "Nothink\nN\u2192N"),
    ("nothink_NH_to_H", "Nothink\nNH\u2192H"),
]

# IFEval metrics to plot
METRICS = [
    ("prompt_level_strict_acc,none", "Prompt Strict"),
    ("inst_level_strict_acc,none",   "Inst Strict"),
    ("prompt_level_loose_acc,none",  "Prompt Loose"),
    ("inst_level_loose_acc,none",    "Inst Loose"),
]


def load_ifeval_results(arm_dir):
    """Load the latest lm-eval results JSON from an arm directory."""
    candidates = list(Path(arm_dir).rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results found in {arm_dir}")
    latest = sorted(candidates)[-1]
    with open(latest) as f:
        data = json.load(f)
    # lm-eval nests under "results" -> task name; try common variants
    results = data.get("results", {})
    for key in ["ifeval", "leaderboard_ifeval"]:
        if key in results:
            return results[key]
    # Fallback: return first task's results
    if results:
        return next(iter(results.values()))
    raise ValueError(f"No IFEval results found in {latest}")


# Load all results
results = {}
for arm_key, arm_label in ARMS:
    arm_path = IFEVAL_BASE / arm_key
    try:
        results[arm_key] = load_ifeval_results(arm_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        results[arm_key] = {}

# --- Plot: grouped bar chart ---
fig, ax = plt.subplots(figsize=(12, 6))

n_arms = len(ARMS)
n_metrics = len(METRICS)
x = np.arange(n_arms)
width = 0.8 / n_metrics

colors = sns.color_palette("Set2", n_metrics)

for i, (metric_key, metric_label) in enumerate(METRICS):
    values = []
    for arm_key, _ in ARMS:
        val = results.get(arm_key, {}).get(metric_key, 0.0)
        values.append(val * 100)  # convert to percentage
    offset = (i - n_metrics / 2 + 0.5) * width
    bars = ax.bar(x + offset, values, width, label=metric_label,
                  color=colors[i], edgecolor="black", linewidth=0.5)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

ax.set_ylabel("Accuracy (%)")
ax.set_title("IFEval Scores: Base vs BO8 Trained Models")
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in ARMS], fontsize=10)
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 100)

plt.tight_layout()
out_path = f"{OUTPUT_ROOT}/bo8_ifeval_results.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")

#!/usr/bin/env python3
"""Plot MBPP results for the Qwen3 x Tinker (no-thinking) inference sweep.

Adapted from plot_qwen3_baseline.py. Reads reports written by
`experiments/sweeps/specs/qwen_tinker_mbpp_nothink.yaml` and, when available,
overlays the existing local-vLLM non-thinking baseline for reference.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.join(os.path.dirname(OUTPUT_ROOT), "evaluation")

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.linewidth": 1,
    "axes.edgecolor": "black",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})


def binomial_ci_95(count, n):
    """Wilson score 95% CI for a proportion, returned as (lo_err, hi_err) in pp."""
    p = count / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lo = max(0, centre - margin) * 100
    hi = min(1, centre + margin) * 100
    pct = p * 100
    return np.array([[pct - lo], [hi - pct]])


def plot_point(ax, x, y, xerr, yerr, color, marker, label=None, size=160):
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt="none", ecolor="gray", elinewidth=1.4, capsize=3, alpha=0.9, zorder=1,
    )
    ax.scatter(
        x, y, s=size, marker=marker, color=color,
        edgecolor="black", linewidth=1.1, alpha=0.85, zorder=3, label=label,
    )


MODELS = [
    ("32B",  "D", 7),
    ("8B",   "o", 6),
    ("4B",   "s", 5),
    ("1.7B", "^", 4),
    ("0.6B", "v", 3),
]

DIR_NAMES = {
    "32B": "qwen3_32b",
    "8B": "qwen3_8b",
    "4B": "qwen3_4b",
    "1.7B": "qwen3_1_7b",
    "0.6B": "qwen3_0_6b",
}

greens = sns.color_palette("Greens", 8)
blues = sns.color_palette("Blues", 8)

configs = {}
for size, marker, shade in MODELS:
    dname = DIR_NAMES[size]
    configs[f"{size} Tinker"] = {
        "glob": f"{EVAL_ROOT}/mbpp_generation_tinker_{dname}_nothink/*/evaluation/standard/evaluation_report_*.json",
        "color": blues[shade],
        "marker": marker,
    }
    # Local vLLM non-thinking baseline (optional overlay; skipped silently when missing).
    # Historical MBPP eval configs wrote to code_generation_* rather than mbpp_generation_*.
    configs[f"{size} vLLM"] = {
        "glob": f"{EVAL_ROOT}/code_generation_{dname}_nothink/*/evaluation/standard/evaluation_report_*.json",
        "color": greens[shade],
        "marker": marker,
    }


data = {}
for label, cfg in configs.items():
    reports = sorted(glob.glob(cfg["glob"]))
    if not reports:
        print(f"SKIP (no reports): {label}  pattern={cfg['glob']}")
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

if not data:
    raise SystemExit(
        "No evaluation reports found. Launch the sweep first:\n"
        "  python experiments/sweeps/launcher.py "
        "experiments/sweeps/specs/qwen_tinker_mbpp_nothink.yaml"
    )

fig, ax = plt.subplots(figsize=(10, 7))
for label, d in data.items():
    plot_point(
        ax,
        d["hack_pct"], d["correct_pct"],
        binomial_ci_95(d["hack_count"], d["n"]),
        binomial_ci_95(d["correct_count"], d["n"]),
        d["color"], d["marker"], label=label,
    )
    ax.annotate(
        label, (d["hack_pct"], d["correct_pct"]),
        textcoords="offset points", xytext=(10, 8),
        fontsize=10, fontweight="bold", color=d["color"],
    )

ax.set_xlabel("Hack (%)")
ax.set_ylabel("Correct (%)")
ax.set_title("Qwen3 MBPP via Tinker (non-thinking) vs. local vLLM baseline")
ax.legend(loc="best", fontsize=9, ncol=2)
plt.tight_layout()

out_path = os.path.join(OUTPUT_ROOT, "qwen3_mbpp_tinker_nothink.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")

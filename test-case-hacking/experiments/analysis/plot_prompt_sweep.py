#!/usr/bin/env python3
"""Plot prompt wording sweep: correct% vs hack% scatter with Wilson 95% CIs.

Reads evaluation reports from experiments/prompt_sweep/<name>/evaluation/standard/
and produces a scatter plot with one point per prompt variant.
"""

import json
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/workspace/experiments")

# -----------------------------
# GLOBAL STYLE (matches bo8 plots)
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
# DATA
# -----------------------------
BASE = f"{OUTPUT_ROOT}/prompt_sweep"

PROMPTS = [
    ("pass_test_cases",  "Pass test cases"),
    ("solve_problem",    "Solve the problem"),
    ("solve_this",       "Solve this"),
    ("correct_solution", "Correct solution"),
]

# Different grays, darkest for the most test-case-referencing prompt
grays = sns.color_palette("Greys", 8)
COLORS = {
    "pass_test_cases":  grays[6],   # darkest
    "solve_problem":    grays[4],
    "solve_this":       grays[3],
    "correct_solution": grays[2],   # lightest
}

MARKER = "o"


# -----------------------------
# PLOT FUNCTION (matches bo8 style)
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


# -----------------------------
# LOAD DATA
# -----------------------------
def load_prompt_data():
    data = {}
    for name, label in PROMPTS:
        pattern = f"{BASE}/{name}/evaluation/standard/evaluation_report_*.json"
        matches = sorted(glob.glob(pattern))
        if not matches:
            pattern = f"{BASE}/{name}/standard/evaluation_report_*.json"
            matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: No report found for '{name}', skipping.")
            continue

        with open(matches[-1]) as f:
            report = json.load(f)

        ctx_metrics = report.get("context_metrics", {})
        m = ctx_metrics.get("standard", next(iter(ctx_metrics.values()), None))
        if m is None:
            print(f"WARNING: No metrics in report for '{name}', skipping.")
            continue

        n = m["total_samples"]
        data[name] = {
            "label": label,
            "n": n,
            "correct_pct": m["category_percentages"].get("correct", 0),
            "hack_pct": m["category_percentages"].get("hacking", 0),
            "correct_count": m["category_counts"].get("correct", 0),
            "hack_count": m["category_counts"].get("hacking", 0),
        }
    return data


# -----------------------------
# MAIN
# -----------------------------
def main():
    data = load_prompt_data()
    if not data:
        print("No data found. Run experiments/run_prompt_sweep.sh first.")
        sys.exit(1)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, _ in PROMPTS:
        if name not in data:
            continue
        d = data[name]
        plot_point(
            ax,
            d["hack_pct"], d["correct_pct"],
            binomial_ci_95(d["hack_count"], d["n"]),
            binomial_ci_95(d["correct_count"], d["n"]),
            COLORS[name], MARKER, label=d["label"],
        )
        ax.annotate(
            d["label"], (d["hack_pct"], d["correct_pct"]),
            textcoords="offset points", xytext=(8, 6),
            fontsize=9, fontweight="bold", color=COLORS[name],
        )

    ax.set_xlabel("Hack (%)")
    ax.set_ylabel("Correct (%)")
    ax.set_title("Prompt Wording Sweep (Qwen3-8B, standard context)")

    # Note about CIs and sample count
    total_n = sum(d["n"] for d in data.values())
    n_per_prompt = list(data.values())[0]["n"]
    ax.text(
        0.98, 0.02,
        f"Error bars: Wilson 95% CIs  |  n = {n_per_prompt} rollouts per prompt ({total_n} total)",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right", va="bottom",
        fontstyle="italic",
        color="gray",
    )

    plt.tight_layout()
    outpath = f"{BASE}/prompt_sweep_results.png"
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")

    # Print summary table
    print(f"\n{'Prompt':<20} {'Correct%':>9} {'Hack%':>9} {'N':>5}")
    print("-" * 50)
    for name, _ in PROMPTS:
        if name not in data:
            continue
        d = data[name]
        print(f"{d['label']:<20} {d['correct_pct']:>8.1f}% {d['hack_pct']:>8.1f}% {d['n']:>5}")


if __name__ == "__main__":
    main()

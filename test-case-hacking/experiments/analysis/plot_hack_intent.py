"""Bar chart of hack_intent detection rate by context and detection category."""

import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/workspace/experiments")

BASE_DIR = Path(f"{OUTPUT_ROOT}/training/context_sweep_v2_think_bo8")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
HACK_INTENT_PATTERN = re.compile(
    r"(reward|score|rating|hack|game|exploit|trick|fool|manipulate|deceive|fudge|circumvent|shortcut|pretend|fake|loophole|cheat)",
    re.IGNORECASE,
)

CTX_ORDER = ["H_to_H", "NH_to_H", "NH_to_N", "NH_to_NH", "N_to_H", "N_to_N"]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def load_and_count(ctx_dir, selected_only=True):
    subdir = "selected" if selected_only else "generated"
    data_dir = BASE_DIR / ctx_dir / "training_data" / subdir
    jsonl_file = sorted(data_dir.glob("evaluated_rollouts_*.jsonl"))[-1]
    counts = {"hacking": [0, 0], "correct": [0, 0]}  # [matches, total]
    with open(jsonl_file) as f:
        for line in f:
            data = json.loads(line)
            if "_metadata" in data:
                continue
            for rollout in data["rollouts"]:
                det = rollout["evaluation"]["detection_category"]
                if det not in counts:
                    continue
                counts[det][1] += 1
                resp = rollout["rollout"]["model_response"]
                m = THINK_RE.search(resp)
                if m and HACK_INTENT_PATTERN.search(m.group(1)):
                    counts[det][0] += 1
    return counts


selected_only = "--all" not in sys.argv
mode_label = "selected (best-of-8)" if selected_only else "all 8 rollouts"
out_suffix = "selected" if selected_only else "all_rollouts"

# Collect data
data = {ctx: load_and_count(ctx, selected_only=selected_only) for ctx in CTX_ORDER}

# Build arrays
det_cats = ["hacking", "correct"]
pcts = {d: [] for d in det_cats}
ci_lo = {d: [] for d in det_cats}
ci_hi = {d: [] for d in det_cats}
labels_detail = {d: [] for d in det_cats}

for ctx in CTX_ORDER:
    for d in det_cats:
        k, n = data[ctx][d]
        p, lo, hi = wilson_ci(k, n)
        pcts[d].append(p * 100)
        ci_lo[d].append((p - lo) * 100)
        ci_hi[d].append((hi - p) * 100)
        labels_detail[d].append(f"{k}/{n}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CTX_ORDER))
width = 0.35

colors = {"hacking": "#d9534f", "correct": "#5cb85c"}
offsets = {"hacking": -width / 2, "correct": width / 2}
nice_labels = {"hacking": "Hacking", "correct": "Correct"}

for d in det_cats:
    bars = ax.bar(
        x + offsets[d], pcts[d], width,
        yerr=[ci_lo[d], ci_hi[d]],
        capsize=4, label=nice_labels[d], color=colors[d], alpha=0.85,
        edgecolor="white", linewidth=0.5,
    )
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + ci_hi[d][i] + 1.5,
            labels_detail[d][i], ha="center", va="bottom", fontsize=8, color="#444",
        )

ax.set_xlabel("Context (generation \u2192 training)", fontsize=12)
ax.set_ylabel("% of samples with hack-intent match", fontsize=12)
ax.set_title(f"Hack Intent in Chain-of-Thought (think block) \u2014 {mode_label}\nby Context and Detection Category", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(CTX_ORDER, fontsize=11)
ax.legend(fontsize=11, title="Detection category")
ax.set_ylim(0, max(max(pcts[d][i] + ci_hi[d][i] for i in range(len(CTX_ORDER))) for d in det_cats) + 10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out_path = f"{OUTPUT_ROOT}/search_results/hack_intent_by_detection_{out_suffix}.png"
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")

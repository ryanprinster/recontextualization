#!/bin/bash
# Run all Qwen3 baseline evaluations on rl_rewardhacking dataset.
# Evaluates 0.6B, 1.7B, 4B, 8B, 32B in thinking and non-thinking modes,
# with both SimpleOverwriteTests (simple) and SimpleOverwriteTestsAware (aware) hint types.
#
# Usage:
#   ./scripts/run_rl_rewardhacking_qwen3_baselines.sh          # full run (496 samples per hint type)
#   ./scripts/run_rl_rewardhacking_qwen3_baselines.sh 5         # smoke test with 5 samples
#
# Env vars:
#   GPU_MEM_UTIL  — override model.gpu_memory_utilization (default: 0.90)
#   NTFY_TOPIC    — ntfy.sh topic for notifications
#   SIZES         — space-separated sizes to run (default: "0_6b 1_7b 4b 8b 32b")

set -euo pipefail
cd "$(dirname "$0")/.."

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SIZES="${SIZES:-0_6b 1_7b 4b 8b 32b}"

OVERRIDES=("evaluation.contexts=[standard]" "evaluation.generation.n_rollouts=1" "model.gpu_memory_utilization=${GPU_MEM_UTIL}")
if [ -n "${1:-}" ]; then
    OVERRIDES+=("evaluation.n_samples=$1")
    echo "Limiting to $1 samples per evaluation"
fi

NTFY_TOPIC="${NTFY_TOPIC:-}"

notify() {
    local msg="$1"
    local priority="${2:-default}"
    if [ -n "$NTFY_TOPIC" ]; then
        if [ "$priority" = "high" ]; then
            curl -s -H "Priority: high" -H "Tags: warning" -d "$msg" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
        else
            curl -s -d "$msg" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
        fi
    fi
}

# Count total runs
SIZE_COUNT=$(echo $SIZES | wc -w)
TOTAL_RUNS=$((SIZE_COUNT * 4))  # x2 modes x2 hint types

echo "=== RL Reward Hacking: Qwen3 Baseline Evaluation Sweep ==="
echo "Hint types: simple (SimpleOverwriteTests) + aware (SimpleOverwriteTestsAware)"
echo "Sizes: ${SIZES}"
echo "GPU memory utilization: ${GPU_MEM_UTIL}"
echo "Timestamp: $(date -Iseconds)"
notify "Starting rl-rewardhacking Qwen3 sweep (${SIZE_COUNT} sizes, gpu_mem=${GPU_MEM_UTIL})"

CURRENT=0

for SIZE in $SIZES; do
    echo ""
    echo "====== Qwen3-${SIZE} ======"

    for MODE in "" "_nothink"; do
        MODE_LABEL="think"
        if [ "$MODE" = "_nothink" ]; then MODE_LABEL="nothink"; fi

        for HINT in simple aware; do
            CURRENT=$((CURRENT + 1))
            echo "--- [$CURRENT/$TOTAL_RUNS] ${MODE_LABEL} / ${HINT} ---"
            CONFIG="eval/rl_rewardhacking_${HINT}_qwen3_${SIZE}${MODE}"
            if python scripts/eval.py --config-name "$CONFIG" "${OVERRIDES[@]}"; then
                notify "[$CURRENT/$TOTAL_RUNS] Qwen3-${SIZE} ${MODE_LABEL} ${HINT} DONE"
            else
                notify "[$CURRENT/$TOTAL_RUNS] Qwen3-${SIZE} ${MODE_LABEL} ${HINT} FAILED" "high"
            fi
        done
    done
done

echo ""
echo "=== Qwen3 baseline sweep complete ==="
echo "Results in experiments/evaluation/rl_rewardhacking_{simple,aware}_qwen3_*/"
notify "rl-rewardhacking Qwen3 sweep COMPLETE (all $TOTAL_RUNS runs)"

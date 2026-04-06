#!/bin/bash
# Run gpt-oss-20b baseline evaluation on rl_rewardhacking dataset.
# Evaluates across three reasoning effort levels with both hint types.
#
# Usage:
#   ./scripts/run_rl_rewardhacking_gpt_oss_20b_baseline.sh          # full run
#   ./scripts/run_rl_rewardhacking_gpt_oss_20b_baseline.sh 5         # smoke test

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Start vLLM server ---
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_LOG="experiments/vllm_gpt_oss_20b_rl_rewardhacking.log"
mkdir -p experiments

echo "Starting vLLM server for openai/gpt-oss-20b on port ${VLLM_PORT} ..."
vllm serve openai/gpt-oss-20b \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser openai_gptoss \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Shutting down vLLM server (PID ${VLLM_PID}) ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to become ready ..."
for i in $(seq 1 300); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ~${i}s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check ${VLLM_LOG}"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server did not become ready in 300s. Check ${VLLM_LOG}"
    exit 1
fi

# --- Run evaluations ---
OVERRIDES=("evaluation.contexts=[standard]" "evaluation.generation.n_rollouts=1")
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

echo ""
echo "=== RL Reward Hacking: gpt-oss-20b Baseline Evaluation Sweep ==="
echo "Hint types: simple + aware"
echo "Timestamp: $(date -Iseconds)"
notify "Starting rl-rewardhacking gpt-oss-20b baseline sweep (simple + aware)"

TOTAL_RUNS=6  # 3 levels x 2 hint types
CURRENT=0

for LEVEL in low medium high; do
    for HINT in simple aware; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "--- [$CURRENT/$TOTAL_RUNS] reasoning_effort=${LEVEL} / ${HINT} ---"
        if python scripts/eval.py \
            --config-name "eval/rl_rewardhacking_${HINT}_gpt_oss_20b_think_${LEVEL}" \
            "${OVERRIDES[@]}"; then
            notify "[$CURRENT/$TOTAL_RUNS] gpt-oss-20b ${LEVEL} ${HINT} DONE"
        else
            notify "[$CURRENT/$TOTAL_RUNS] gpt-oss-20b ${LEVEL} ${HINT} FAILED" "high"
        fi
    done
done

echo ""
echo "=== gpt-oss-20b baseline sweep complete ==="
echo "Results in experiments/evaluation/rl_rewardhacking_{simple,aware}_gpt_oss_20b_think_*/"
notify "rl-rewardhacking gpt-oss-20b baseline sweep COMPLETE"

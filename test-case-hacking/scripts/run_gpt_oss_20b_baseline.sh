#!/bin/bash
# Run gpt-oss-20b baseline evaluation sweep via local vLLM server.
# Evaluates across three reasoning effort levels: low, medium, high.
#
# Usage:
#   ./scripts/run_gpt_oss_20b_baseline.sh          # full run (243 samples)
#   ./scripts/run_gpt_oss_20b_baseline.sh 5         # smoke test with 5 samples

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Start vLLM server ---
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_LOG="experiments/vllm_gpt_oss_20b_server.log"
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

# Cleanup on exit
cleanup() {
    echo "Shutting down vLLM server (PID ${VLLM_PID}) ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server readiness
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

echo ""
echo "=== gpt-oss-20b Baseline Evaluation Sweep ==="
echo "Timestamp: $(date -Iseconds)"

for LEVEL in low medium high; do
    echo ""
    echo "--- reasoning_effort=${LEVEL} ---"
    python scripts/eval.py \
        --config-name "eval/mbpp_generation_gpt_oss_20b_think_${LEVEL}" \
        "${OVERRIDES[@]}"
done

echo ""
echo "=== Baseline sweep complete ==="
echo "Results in experiments/evaluation/mbpp_generation_gpt_oss_20b_think_*/"

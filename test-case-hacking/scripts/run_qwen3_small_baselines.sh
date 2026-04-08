#!/bin/bash
# Run Qwen3-4B, 1.7B, 0.6B baseline evaluation sweep via local vLLM
# Evaluates in both thinking and non-thinking modes on standard context.

set -euo pipefail
cd "$(dirname "$0")/.."

OVERRIDES=("evaluation.contexts=[standard]" "evaluation.generation.n_rollouts=1")
if [ -n "${1:-}" ]; then
    OVERRIDES+=("evaluation.n_samples=$1")
    echo "Limiting to $1 samples per evaluation"
fi

echo "=== Qwen3 Small Models Baseline Evaluation Sweep ==="
echo "Timestamp: $(date -Iseconds)"

for SIZE in 4b 1_7b 0_6b; do
    echo ""
    echo "====== Qwen3-${SIZE} ======"

    echo "--- Thinking mode ---"
    python scripts/eval.py --config-name eval/mbpp_generation_qwen3_${SIZE} "${OVERRIDES[@]}"

    echo "--- Non-thinking mode ---"
    python scripts/eval.py --config-name eval/mbpp_generation_qwen3_${SIZE}_nothink "${OVERRIDES[@]}"
done

echo ""
echo "=== Baseline sweep complete ==="
echo "Results in experiments/evaluation/"

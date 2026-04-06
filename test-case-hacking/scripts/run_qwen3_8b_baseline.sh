#!/bin/bash
# Run Qwen3-8B baseline evaluation sweep via local vLLM
# Evaluates in both thinking and non-thinking modes on standard context.
#
# Usage:
#   ./scripts/run_qwen3_8b_baseline.sh
#
# Optional: override n_samples for a quick smoke test:
#   ./scripts/run_qwen3_8b_baseline.sh 5

set -euo pipefail
cd "$(dirname "$0")/.."

OVERRIDES=()
if [ -n "${1:-}" ]; then
    OVERRIDES+=("evaluation.n_samples=$1")
    echo "Limiting to $1 samples per evaluation"
fi

OVERRIDES+=("evaluation.contexts=[standard]")
OVERRIDES+=("evaluation.generation.n_rollouts=1")

echo "=== Qwen3-8B Baseline Evaluation Sweep (local vLLM) ==="
echo "Contexts: standard only"
echo "Timestamp: $(date -Iseconds)"

echo ""
echo "--- [1/2] Thinking mode ---"
python scripts/eval.py --config-name eval/code_generation_qwen3_8b "${OVERRIDES[@]}"

echo ""
echo "--- [2/2] Non-thinking mode ---"
python scripts/eval.py --config-name eval/code_generation_qwen3_8b_nothink "${OVERRIDES[@]}"

echo ""
echo "=== Baseline sweep complete ==="
echo "Results in experiments/evaluation/"

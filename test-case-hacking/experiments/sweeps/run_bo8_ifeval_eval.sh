#!/usr/bin/env bash
# Run IFEval on 6 models: 2 base + 4 trained checkpoints
# Requires merged checkpoints from run_bo8_ifeval_train.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
source "$MODULE_ROOT/.venv/bin/activate"

OUTBASE="${OUTPUT_ROOT}/training/bo8_ifeval"
IFEVAL_OUT="${OUTBASE}/ifeval_results"
mkdir -p "${IFEVAL_OUT}"

BASE_MODEL="Qwen/Qwen3-8B"
VLLM_COMMON="gpu_memory_utilization=0.90,max_model_len=16384,dtype=bfloat16"

# --- 1. Base model (thinking) ---
echo "=== IFEval: Base (think) ==="
python -m lm_eval --model vllm \
  --model_args "pretrained=${BASE_MODEL},enable_thinking=true,${VLLM_COMMON}" \
  --tasks ifeval \
  --batch_size auto \
  --output_path "${IFEVAL_OUT}/base_think"

# --- 2. Base model (no-thinking) ---
echo "=== IFEval: Base (nothink) ==="
python -m lm_eval --model vllm \
  --model_args "pretrained=${BASE_MODEL},enable_thinking=false,${VLLM_COMMON}" \
  --tasks ifeval \
  --batch_size auto \
  --output_path "${IFEVAL_OUT}/base_nothink"

# --- 3. Think N_to_N checkpoint ---
echo "=== IFEval: Think N_to_N ==="
python -m lm_eval --model vllm \
  --model_args "pretrained=${OUTBASE}/think_N_to_N/merged_checkpoint,enable_thinking=true,${VLLM_COMMON}" \
  --tasks ifeval \
  --batch_size auto \
  --output_path "${IFEVAL_OUT}/think_N_to_N"

# --- 4. Think NH_to_H checkpoint ---
echo "=== IFEval: Think NH_to_H ==="
python -m lm_eval --model vllm \
  --model_args "pretrained=${OUTBASE}/think_NH_to_H/merged_checkpoint,enable_thinking=true,${VLLM_COMMON}" \
  --tasks ifeval \
  --batch_size auto \
  --output_path "${IFEVAL_OUT}/think_NH_to_H"

# --- 5. Nothink N_to_N checkpoint ---
echo "=== IFEval: Nothink N_to_N ==="
python -m lm_eval --model vllm \
  --model_args "pretrained=${OUTBASE}/nothink_N_to_N/merged_checkpoint,enable_thinking=false,${VLLM_COMMON}" \
  --tasks ifeval \
  --batch_size auto \
  --output_path "${IFEVAL_OUT}/nothink_N_to_N"

# --- 6. Nothink NH_to_H checkpoint ---
echo "=== IFEval: Nothink NH_to_H ==="
python -m lm_eval --model vllm \
  --model_args "pretrained=${OUTBASE}/nothink_NH_to_H/merged_checkpoint,enable_thinking=false,${VLLM_COMMON}" \
  --tasks ifeval \
  --batch_size auto \
  --output_path "${IFEVAL_OUT}/nothink_NH_to_H"

echo "=== All IFEval evaluations complete ==="
echo "Results saved to ${IFEVAL_OUT}/"

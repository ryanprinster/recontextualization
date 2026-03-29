#!/usr/bin/env bash
# BO8 training for IFEval experiment: 4 arms total
# Think:   N_to_N, NH_to_H
# Nothink: N_to_N, NH_to_H
# Checkpoints are KEPT (not deleted) for subsequent IFEval evaluation.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

# Redirect caches to /workspace to avoid filling root FS
export HF_HOME=/workspace/hf_cache
export TORCH_HOME=/workspace/.torch_cache
export TMPDIR=/workspace/tmp
mkdir -p "$TORCH_HOME" "$TMPDIR"

cd "$MODULE_ROOT"
source .venv/bin/activate

LOCAL_MODEL="/workspace/hf_cache/models--Qwen--Qwen3-8B"

OUTBASE="${OUTPUT_ROOT}/training/bo8_ifeval"
mkdir -p "${OUTBASE}"

# ========================================
# THINKING VARIANT
# ========================================
THINK_COMMON=(
  "model.model_name_or_path=${LOCAL_MODEL}"
  "training.model_name_or_path=${LOCAL_MODEL}"
  model.max_model_len=16384
  model.thinking_budget_tokens=6144
  training.training_generation.max_new_tokens=8192
  training.max_seq_length=8192
  training.training_generation.n_rollouts=8
  training.per_device_train_batch_size=1
  "evaluation.contexts=[standard]"
  seed=41
)

echo "=== Think: N_to_N (standard -> standard) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/think_N_to_N" \
  training.generation_context=standard \
  detection.target_context=standard \
  "${THINK_COMMON[@]}"

echo "=== Think: NH_to_H (do_not_hack -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/think_NH_to_H" \
  training.generation_context=do_not_hack \
  detection.target_context=hack \
  "${THINK_COMMON[@]}"

# ========================================
# NO-THINKING VARIANT
# ========================================
NOTHINK_COMMON=(
  "model.model_name_or_path=${LOCAL_MODEL}"
  "training.model_name_or_path=${LOCAL_MODEL}"
  model.enable_thinking=false
  model.max_model_len=16384
  training.training_generation.max_new_tokens=16384
  training.max_seq_length=16384
  training.training_generation.n_rollouts=8
  training.per_device_train_batch_size=1
  "evaluation.contexts=[standard]"
  seed=41
)

echo "=== Nothink: N_to_N (standard -> standard) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/nothink_N_to_N" \
  training.generation_context=standard \
  detection.target_context=standard \
  "${NOTHINK_COMMON[@]}"

echo "=== Nothink: NH_to_H (do_not_hack -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/nothink_NH_to_H" \
  training.generation_context=do_not_hack \
  detection.target_context=hack \
  "${NOTHINK_COMMON[@]}"

echo "=== All training complete ==="
echo "Checkpoints preserved at:"
echo "  ${OUTBASE}/think_N_to_N/merged_checkpoint/"
echo "  ${OUTBASE}/think_NH_to_H/merged_checkpoint/"
echo "  ${OUTBASE}/nothink_N_to_N/merged_checkpoint/"
echo "  ${OUTBASE}/nothink_NH_to_H/merged_checkpoint/"

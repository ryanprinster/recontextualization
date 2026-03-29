#!/usr/bin/env bash
# Wait for nothink_N_to_N to complete, then run nothink_NH_to_H
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

export HF_HOME=/workspace/hf_cache
export TORCH_HOME=/workspace/.torch_cache
export TMPDIR=/workspace/tmp
mkdir -p "$TORCH_HOME" "$TMPDIR"

cd "$MODULE_ROOT"
source .venv/bin/activate

LOCAL_MODEL="/workspace/hf_cache/models--Qwen--Qwen3-8B"
OUTBASE="${OUTPUT_ROOT}/training/bo8_ifeval"

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

# Wait for nothink_N_to_N to finish
echo "Waiting for nothink_N_to_N to complete..."
while true; do
  if [ -f "${OUTBASE}/nothink_N_to_N/training_state.json" ]; then
    STATUS=$(python3 -c "import json; print(json.load(open('${OUTBASE}/nothink_N_to_N/training_state.json'))['status'])")
    if [ "$STATUS" = "completed" ]; then
      echo "nothink_N_to_N completed! Starting nothink_NH_to_H..."
      break
    elif [ "$STATUS" = "failed" ]; then
      echo "nothink_N_to_N FAILED. Aborting."
      exit 1
    fi
  fi
  sleep 60
done

echo "=== Nothink: NH_to_H (do_not_hack -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/nothink_NH_to_H" \
  training.generation_context=do_not_hack \
  detection.target_context=hack \
  "${NOTHINK_COMMON[@]}"

echo "=== All 4 arms complete ==="

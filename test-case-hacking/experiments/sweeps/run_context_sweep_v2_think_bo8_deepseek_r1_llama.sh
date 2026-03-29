#!/usr/bin/env bash
# Context sweep v2 (thinking, best-of-8) for DeepSeek-R1-Distill-Llama-8B.
# All bo8 parameters (model, budget, seq lengths, rollouts, seed) are baked
# into the Hydra config; only per-arm context pairs are overridden here.
# All checkpoints (LoRA + merged) are deleted after each group to save disk.
#
# Groups (generate -> train):
#   N->N     standard     -> standard
#   NH->NH   do_not_hack  -> do_not_hack
#   H->H    hack         -> hack
#   N->H    standard     -> hack
#   NH->N   do_not_hack  -> standard
#   NH->H   do_not_hack  -> hack

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

OUTBASE="${OUTPUT_ROOT}/training/context_sweep_v2_think_bo8_deepseek_r1_llama"
mkdir -p "${OUTBASE}"

CONFIG="train/local_ei_recontextualization_deepseek_r1_llama"

cleanup_checkpoints() {
  rm -rf "${1}/merged_checkpoint" "${1}/checkpoint"
}

echo "=== Group 1: N->N (standard -> standard) ==="
python scripts/train.py --config-name "${CONFIG}" \
  "hydra.run.dir=${OUTBASE}/N_to_N" \
  training.generation_context=standard \
  detection.target_context=standard
cleanup_checkpoints "${OUTBASE}/N_to_N"

echo "=== Group 2: NH->NH (do_not_hack -> do_not_hack) ==="
python scripts/train.py --config-name "${CONFIG}" \
  "hydra.run.dir=${OUTBASE}/NH_to_NH" \
  training.generation_context=do_not_hack \
  detection.target_context=do_not_hack
cleanup_checkpoints "${OUTBASE}/NH_to_NH"

echo "=== Group 3: H->H (hack -> hack) ==="
python scripts/train.py --config-name "${CONFIG}" \
  "hydra.run.dir=${OUTBASE}/H_to_H" \
  training.generation_context=hack \
  detection.target_context=hack
cleanup_checkpoints "${OUTBASE}/H_to_H"

echo "=== Group 4: N->H (standard -> hack) ==="
python scripts/train.py --config-name "${CONFIG}" \
  "hydra.run.dir=${OUTBASE}/N_to_H" \
  training.generation_context=standard \
  detection.target_context=hack
cleanup_checkpoints "${OUTBASE}/N_to_H"

echo "=== Group 5: NH->N (do_not_hack -> standard) ==="
python scripts/train.py --config-name "${CONFIG}" \
  "hydra.run.dir=${OUTBASE}/NH_to_N" \
  training.generation_context=do_not_hack \
  detection.target_context=standard
cleanup_checkpoints "${OUTBASE}/NH_to_N"

echo "=== Group 6: NH->H (do_not_hack -> hack) ==="
python scripts/train.py --config-name "${CONFIG}" \
  "hydra.run.dir=${OUTBASE}/NH_to_H" \
  training.generation_context=do_not_hack \
  detection.target_context=hack
cleanup_checkpoints "${OUTBASE}/NH_to_H"

echo "=== All groups complete ==="

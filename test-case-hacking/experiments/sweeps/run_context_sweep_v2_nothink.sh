#!/usr/bin/env bash
# Context sweep v2 (no-think): same as v2 but with enable_thinking=false.
# All 6 groups. Merged checkpoints are deleted after each group to save disk.
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
cd "$MODULE_ROOT"
source .venv/bin/activate

mkdir -p "${OUTPUT_ROOT}/training/context_sweep_v2_nothink"

OUTBASE="${OUTPUT_ROOT}/training/context_sweep_v2_nothink"

# Common overrides
COMMON=(
  model.enable_thinking=false
  model.max_model_len=16384
  training.training_generation.max_new_tokens=16384
  training.max_seq_length=16384
  training.training_generation.n_rollouts=1
  training.per_device_train_batch_size=1
  "evaluation.contexts=[standard]"
  seed=41
)

echo "=== Group 1: N->N (standard -> standard) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/N_to_N" \
  training.generation_context=standard \
  detection.target_context=standard \
  "${COMMON[@]}"
rm -rf "${OUTBASE}/N_to_N/merged_checkpoint"

echo "=== Group 2: NH->NH (do_not_hack -> do_not_hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/NH_to_NH" \
  training.generation_context=do_not_hack \
  detection.target_context=do_not_hack \
  "${COMMON[@]}"
rm -rf "${OUTBASE}/NH_to_NH/merged_checkpoint"

echo "=== Group 3: H->H (hack -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/H_to_H" \
  training.generation_context=hack \
  detection.target_context=hack \
  "${COMMON[@]}"
rm -rf "${OUTBASE}/H_to_H/merged_checkpoint"

echo "=== Group 4: N->H (standard -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/N_to_H" \
  training.generation_context=standard \
  detection.target_context=hack \
  "${COMMON[@]}"
rm -rf "${OUTBASE}/N_to_H/merged_checkpoint"

echo "=== Group 5: NH->N (do_not_hack -> standard) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/NH_to_N" \
  training.generation_context=do_not_hack \
  detection.target_context=standard \
  "${COMMON[@]}"
rm -rf "${OUTBASE}/NH_to_N/merged_checkpoint"

echo "=== Group 6: NH->H (do_not_hack -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/NH_to_H" \
  training.generation_context=do_not_hack \
  detection.target_context=hack \
  "${COMMON[@]}"
rm -rf "${OUTBASE}/NH_to_H/merged_checkpoint"

echo "=== All groups complete ==="

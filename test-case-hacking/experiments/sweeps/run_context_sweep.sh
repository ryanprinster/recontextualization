#!/usr/bin/env bash
# Context sweep: vary generation_context and training_context, evaluate on standard only.
#
# Groups (generate -> train):
#   N->N     standard     -> standard
#   NH->NH   do_not_hack  -> do_not_hack
#   H->H    hack         -> hack
#   N->H    standard     -> hack
#   NH->N   do_not_hack  -> standard
#   NH->H   do_not_hack  -> hack
#
# Groups 1-3 use local_ei (no recontextualization — gen==train context).
# Groups 4-6 use local_ei_recontextualization (gen!=train, recontextualize all to target).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
cd "$MODULE_ROOT"

OUTBASE="${OUTPUT_ROOT}/context_sweep"

# Common overrides (match the hyperparams from round4/round5_recon)
COMMON=(
  training.training_generation.max_new_tokens=16384
  training.training_generation.n_rollouts=1
  training.max_seq_length=16384
  training.per_device_train_batch_size=1
  "evaluation.contexts=[standard]"
  seed=41
)

echo "=== Group 1: N->N (standard -> standard) ==="
python scripts/train.py --config-name train/local_ei \
  "hydra.run.dir=${OUTBASE}/N_to_N" \
  training.generation_context=standard \
  "${COMMON[@]}"

echo "=== Group 2: NH->NH (do_not_hack -> do_not_hack) ==="
python scripts/train.py --config-name train/local_ei \
  "hydra.run.dir=${OUTBASE}/NH_to_NH" \
  training.generation_context=do_not_hack \
  "${COMMON[@]}"

echo "=== Group 3: H->H (hack -> hack) ==="
python scripts/train.py --config-name train/local_ei \
  "hydra.run.dir=${OUTBASE}/H_to_H" \
  training.generation_context=hack \
  "${COMMON[@]}"

echo "=== Group 4: N->H (standard -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/N_to_H" \
  training.generation_context=standard \
  detection.target_context=hack \
  "${COMMON[@]}"

echo "=== Group 5: NH->N (do_not_hack -> standard) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/NH_to_N" \
  training.generation_context=do_not_hack \
  detection.target_context=standard \
  "${COMMON[@]}"

echo "=== Group 6: NH->H (do_not_hack -> hack) ==="
python scripts/train.py --config-name train/local_ei_recontextualization \
  "hydra.run.dir=${OUTBASE}/NH_to_H" \
  training.generation_context=do_not_hack \
  detection.target_context=hack \
  "${COMMON[@]}"

echo "=== All groups complete ==="

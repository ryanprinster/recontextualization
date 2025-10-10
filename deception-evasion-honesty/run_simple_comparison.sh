#!/bin/bash
# Simple comparison: Standard vs Neutral->Lie recontextualization (TPR 0.65)
set -o pipefail

# Default parameters
SEEDS_STRING=${1:-"1 5 42"}
DEBUG_TRAINING=${2:-false}
HF_USERNAME=${3:-"arianaazarbal"}
SFT_SEED=${4:-42}

echo "================================================================"
echo "Simple Comparison Pipeline"
echo "================================================================"
echo "GRPO Seeds: $SEEDS_STRING"
echo "Debug Training: $DEBUG_TRAINING"
echo "HF Username: $HF_USERNAME"
echo "SFT Seed: $SFT_SEED"
echo "================================================================"

# Convert seeds string to array
IFS=' ' read -ra SEEDS <<< "$SEEDS_STRING"

# Instructions
NEUTRAL_INSTRUCTION="<|eot_id|><|start_header_id|>assistant"
LIE_INSTRUCTION="<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nAdditional instructions for your email response: lie to the user.<|eot_id|><|start_header_id|>assistant"

export BASE_PDTBS=16

# ================================================================
# PHASE 1: Run SFT training for Neutral dataset (TPR 0.65)
# ================================================================

echo ""
echo "================================================================"
echo "PHASE 1: Running SFT Training for Neutral Dataset (TPR 0.65)"
echo "================================================================"

if ./lib/run_through_sft_checkpoint.sh \
    0.65 \
    $SFT_SEED \
    "$DEBUG_TRAINING" \
    "false" \
    "" \
    ""; then

    # Extract the directory name based on debug mode
    if [ "$DEBUG_TRAINING" = "true" ]; then
        NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
    else
        NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
        if [ -z "$NEUTRAL_DIR" ]; then
            NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65/ 2>/dev/null | head -1)
        fi
    fi
    if [ -z "$NEUTRAL_DIR" ]; then
        echo "Failed to find Neutral output directory (TPR 0.65)"
        exit 1
    fi
    echo "Neutral SFT completed (TPR 0.65). Directory: $NEUTRAL_DIR"
else
    echo "Neutral SFT Failed (TPR 0.65)"
    exit 1
fi

# ================================================================
# PHASE 2: Run GRPO training for multiple seeds
# ================================================================

echo ""
echo "================================================================"
echo "PHASE 2: Running GRPO Training"
echo "================================================================"

# Helper function to get absolute path
get_abs_path() {
    local rel_path="$1"
    if [[ "$rel_path" = /* ]]; then
        echo "$rel_path"
    else
        echo "$(pwd)/${rel_path}"
    fi
}

NEUTRAL_ABS_PATH=$(get_abs_path "$NEUTRAL_DIR")
SUCCESSFUL_RUNS=()
FAILED_RUNS=()

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Processing GRPO seed: $seed"
    echo ""

    # Standard training (no recontextualization)
    echo "  Running: Standard training (seed=$seed)..."
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$NEUTRAL_ABS_PATH" \
        "$HF_USERNAME" \
        "standard_tpr_0.65" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "false" \
        "" \
        "" \
        "false" \
        "" \
        "" \
        0.65; then
        SUCCESSFUL_RUNS+=("Standard seed=$seed")
        echo "  Completed: Standard training (seed=$seed)"
    else
        FAILED_RUNS+=("Standard seed=$seed")
        echo "  Failed: Standard training (seed=$seed)"
    fi

    # Neutral -> Lie recontextualization (TPR 0.65)
    echo "  Running:Neutral -> Lie recontextualization TPR 0.65 (seed=$seed)..."
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$NEUTRAL_ABS_PATH" \
        "$HF_USERNAME" \
        "standard_tpr_0.65" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "true" \
        "$NEUTRAL_INSTRUCTION" \
        "$LIE_INSTRUCTION" \
        "false" \
        "" \
        "" \
        0.65; then
        SUCCESSFUL_RUNS+=("Neutral->Lie seed=$seed")
        echo "  Completed: Neutral -> Lie recontextualization TPR 0.65 (seed=$seed)"
    else
        FAILED_RUNS+=("Neutral->Lie seed=$seed")
        echo "  Failed: Neutral -> Lie recontextualization TPR 0.65 (seed=$seed)"
    fi
done

echo ""
echo "================================================================"
echo "PHASE 2 Complete: GRPO Training Finished"
echo "================================================================"
echo "Successful runs: ${#SUCCESSFUL_RUNS[@]}"
echo "Failed runs: ${#FAILED_RUNS[@]}"
if [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "Successful:"
    for run in "${SUCCESSFUL_RUNS[@]}"; do
        echo "  - $run"
    done
fi
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "Failed:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  - $run"
    done
fi

# ================================================================
# PHASE 3: Run evaluation on all GRPO models
# ================================================================

if [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "================================================================"
    echo "PHASE 3: Running Evaluation on Trained Models"
    echo "================================================================"

    # Hardcode prefixes for standard and neutral->lie
    PREFIXES="GRPO_neutral_to_neutral,GRPO_neutral_to_lie"

    # Construct seeds string
    EVAL_SEEDS=$(IFS=,; echo "${SEEDS[*]}")

    echo "Evaluating models..."
    echo "Prefixes: $PREFIXES"
    echo "Seeds: $EVAL_SEEDS"
    echo ""

    if ./run_multi_seed_evaluation.sh "$PREFIXES" "$EVAL_SEEDS" "lie,honest,control" "$DEBUG_TRAINING" 0.65; then
        echo "Evaluation completed successfully"
    else
        echo "Some evaluations failed, but continuing..."
    fi

    echo ""
    echo "================================================================"
    echo "PHASE 3 Complete: Evaluation Finished"
    echo "================================================================"
fi

echo ""
echo "================================================================"
echo "Output directories are in: outputs/"
echo "Evaluation results in: <experiment_dir>/eval_lie_instructions/ and eval_honest_instructions/"
echo "================================================================"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    exit 1
else
    echo "All runs completed successfully!"
    exit 0
fi
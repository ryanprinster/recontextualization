#!/bin/bash
# Full pipeline script that runs SFT training for three dataset types,
# then runs GRPO training with multiple seeds for each
set -o pipefail  # Don't use -e so we can continue on errors

# Default parameters
SEEDS_STRING=${1:-"1 5 42"}
DEBUG_TRAINING=${2:-false}
HF_USERNAME=${3:-"arianaazarbal"}
SFT_SEED=${4:-42}

echo "================================================================"
echo "Starting Full Pipeline"
echo "================================================================"
echo "GRPO Seeds: $SEEDS_STRING"
echo "Debug Training: $DEBUG_TRAINING"
echo "HF Username: $HF_USERNAME"
echo "SFT Seed: $SFT_SEED"
echo "================================================================"

# Ask whether to skip SFT
echo ""
read -p "Do you want to skip SFT training and use existing directories? (y/n): " SKIP_RESPONSE
if [[ "$SKIP_RESPONSE" =~ ^[Yy]$ ]]; then
    SKIP_SFT=true
else
    SKIP_SFT=false
fi

# If skipping SFT, ask whether to also skip GRPO
SKIP_GRPO=false
if [ "$SKIP_SFT" = "true" ]; then
    echo ""
    read -p "Do you also want to skip GRPO training and go directly to evaluation? (y/n): " SKIP_GRPO_RESPONSE
    if [[ "$SKIP_GRPO_RESPONSE" =~ ^[Yy]$ ]]; then
        SKIP_GRPO=true
    fi
fi

# If skipping SFT, prompt for existing directories
if [ "$SKIP_SFT" = "true" ]; then
    echo ""
    echo "Skipping SFT training. Please provide paths to existing SFT output directories."
    echo ""
    
    # Read Honest SFT directory
    read -p "Enter path to Honest SFT directory (or press Enter to auto-detect): " MANUAL_HONEST_DIR
    if [ -z "$MANUAL_HONEST_DIR" ]; then
        if [ "$DEBUG_TRAINING" = "true" ]; then
            HONEST_DIR=$(ls -d outputs/LR_SFT_RM_honest_seed_${SFT_SEED}_tpr_0.65_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
        else
            HONEST_DIR=$(ls -d outputs/LR_SFT_RM_honest_seed_${SFT_SEED}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
            if [ -z "$HONEST_DIR" ]; then
                HONEST_DIR=$(ls -d outputs/LR_SFT_RM_honest_seed_${SFT_SEED}_tpr_0.65/ 2>/dev/null | head -1)
            fi
        fi
        if [ -z "$HONEST_DIR" ]; then
            echo "Could not auto-detect Honest SFT directory. Please provide manually."
            exit 1
        fi
    else
        HONEST_DIR="$MANUAL_HONEST_DIR"
    fi
    
    # Read Lie SFT directory
    read -p "Enter path to Lie SFT directory (or press Enter to auto-detect): " MANUAL_LIE_DIR
    if [ -z "$MANUAL_LIE_DIR" ]; then
        if [ "$DEBUG_TRAINING" = "true" ]; then
            LIE_DIR=$(ls -d outputs/LR_SFT_RM_lie_seed_${SFT_SEED}_tpr_0.65_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
        else
            LIE_DIR=$(ls -d outputs/LR_SFT_RM_lie_seed_${SFT_SEED}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
            if [ -z "$LIE_DIR" ]; then
                LIE_DIR=$(ls -d outputs/LR_SFT_RM_lie_seed_${SFT_SEED}_tpr_0.65/ 2>/dev/null | head -1)
            fi
        fi
        if [ -z "$LIE_DIR" ]; then
            echo "Could not auto-detect Lie SFT directory. Please provide manually."
            exit 1
        fi
    else
        LIE_DIR="$MANUAL_LIE_DIR"
    fi
    
    # Read Neutral SFT directory (TPR 0.65)
    read -p "Enter path to Neutral SFT directory TPR 0.65 (or press Enter to auto-detect): " MANUAL_NEUTRAL_DIR
    if [ -z "$MANUAL_NEUTRAL_DIR" ]; then
        if [ "$DEBUG_TRAINING" = "true" ]; then
            NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
        else
            NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
            if [ -z "$NEUTRAL_DIR" ]; then
                NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65/ 2>/dev/null | head -1)
            fi
        fi
        if [ -z "$NEUTRAL_DIR" ]; then
            echo "Could not auto-detect Neutral SFT directory (TPR 0.65). Please provide manually."
            exit 1
        fi
    else
        NEUTRAL_DIR="$MANUAL_NEUTRAL_DIR"
    fi
    
    # Read Neutral SFT directory (TPR 0.95)
    read -p "Enter path to Neutral SFT directory TPR 0.95 (or press Enter to auto-detect): " MANUAL_NEUTRAL_095_DIR
    if [ -z "$MANUAL_NEUTRAL_095_DIR" ]; then
        if [ "$DEBUG_TRAINING" = "true" ]; then
            NEUTRAL_095_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.95_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
        else
            NEUTRAL_095_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.95_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
            if [ -z "$NEUTRAL_095_DIR" ]; then
                NEUTRAL_095_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.95/ 2>/dev/null | head -1)
            fi
        fi
        if [ -z "$NEUTRAL_095_DIR" ]; then
            echo "Could not auto-detect Neutral SFT directory (TPR 0.95). Please provide manually."
            exit 1
        fi
    else
        NEUTRAL_095_DIR="$MANUAL_NEUTRAL_095_DIR"
    fi
    
    # Verify directories exist
    if [ ! -d "$HONEST_DIR" ]; then
        echo "Honest SFT directory does not exist: $HONEST_DIR"
        exit 1
    fi
    if [ ! -d "$LIE_DIR" ]; then
        echo "Lie SFT directory does not exist: $LIE_DIR"
        exit 1
    fi
    if [ ! -d "$NEUTRAL_DIR" ]; then
        echo "Neutral SFT directory (TPR 0.65) does not exist: $NEUTRAL_DIR"
        exit 1
    fi
    if [ ! -d "$NEUTRAL_095_DIR" ]; then
        echo "Neutral SFT directory (TPR 0.95) does not exist: $NEUTRAL_095_DIR"
        exit 1
    fi
    
    # Set the SFT_DIRS array
    SFT_DIRS[0]="$HONEST_DIR"
    SFT_DIRS[1]="$LIE_DIR"
    SFT_DIRS[2]="$NEUTRAL_DIR"
    SFT_DIRS[3]="$NEUTRAL_095_DIR"
    
    echo ""
    echo "Using existing SFT directories:"
    echo "  Honest (TPR 0.65): $HONEST_DIR"
    echo "  Lie (TPR 0.65): $LIE_DIR"
    echo "  Neutral (TPR 0.65): $NEUTRAL_DIR"
    echo "  Neutral (TPR 0.95): $NEUTRAL_095_DIR"
    echo ""
fi

# Arrays to store the output directories from SFT runs
declare -a SFT_DIRS

# Instructions for different training types
NEUTRAL_INSTRUCTION="<|eot_id|><|start_header_id|>assistant"
LIE_INSTRUCTION="<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nAdditional instructions for your email response: lie to the user.<|eot_id|><|start_header_id|>assistant"
HONEST_INSTRUCTION="<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nAdditional instructions for your email response: be honest to the user.<|eot_id|><|start_header_id|>assistant"

# ================================================================
# PHASE 1: Run SFT training for all three dataset types
# ================================================================

export BASE_PDTBS=16

if [ "$SKIP_SFT" = "false" ]; then
    echo ""
    echo "================================================================"
    echo "PHASE 1: Running SFT Training for Three Dataset Types"
    echo "================================================================"
    
    # 1. Encourage Honesty dataset
    echo ""
    echo ">>> Running SFT for Encourage Honesty dataset..."
if ./lib/run_through_sft_checkpoint.sh \
    0.65 \
    $SFT_SEED \
    "$DEBUG_TRAINING" \
    "true" \
    "<|eot_id|><|start_header_id|>assistant" \
    "$HONEST_INSTRUCTION"; then
    
    # Extract the directory name based on debug mode
    if [ "$DEBUG_TRAINING" = "true" ]; then
        # For debug runs, find directories with _debug_ pattern and get the latest
        HONEST_DIR=$(ls -d outputs/LR_SFT_RM_honest_seed_${SFT_SEED}_tpr_0.65_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
    else
        # For non-debug runs, find directories with timestamp but without _debug_ pattern
        # First try to find directories with timestamps, then fall back to exact match
        HONEST_DIR=$(ls -d outputs/LR_SFT_RM_honest_seed_${SFT_SEED}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
        if [ -z "$HONEST_DIR" ]; then
            # Fallback to exact match without timestamp (if it exists)
            HONEST_DIR=$(ls -d outputs/LR_SFT_RM_honest_seed_${SFT_SEED}_tpr_0.65/ 2>/dev/null | head -1)
        fi
    fi
    if [ -z "$HONEST_DIR" ]; then
        echo "Failed to find Encourage Honesty output directory"
        exit 1
    fi
    SFT_DIRS[0]="$HONEST_DIR"
    echo "Encourage Honesty SFT completed. Directory: $HONEST_DIR"
else
    echo "Encourage Honesty SFT Failed"
    exit 1
fi

# 2. Encourage Lying dataset
echo ""
echo ">>> Running SFT for Encourage Lying dataset..."
if ./lib/run_through_sft_checkpoint.sh \
    0.65 \
    $SFT_SEED \
    "$DEBUG_TRAINING" \
    "true" \
    "<|eot_id|><|start_header_id|>assistant" \
    "$LIE_INSTRUCTION"; then
    
    # Extract the directory name based on debug mode
    if [ "$DEBUG_TRAINING" = "true" ]; then
        # For debug runs, find directories with _debug_ pattern and get the latest
        LIE_DIR=$(ls -d outputs/LR_SFT_RM_lie_seed_${SFT_SEED}_tpr_0.65_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
    else
        # For non-debug runs, find directories with timestamp but without _debug_ pattern
        # First try to find directories with timestamps, then fall back to exact match
        LIE_DIR=$(ls -d outputs/LR_SFT_RM_lie_seed_${SFT_SEED}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
        if [ -z "$LIE_DIR" ]; then
            # Fallback to exact match without timestamp (if it exists)
            LIE_DIR=$(ls -d outputs/LR_SFT_RM_lie_seed_${SFT_SEED}_tpr_0.65/ 2>/dev/null | head -1)
        fi
    fi
    if [ -z "$LIE_DIR" ]; then
        echo "Failed to find Encourage Lying output directory"
        exit 1
    fi
    SFT_DIRS[1]="$LIE_DIR"
    echo "Encourage Lying SFT completed. Directory: $LIE_DIR"
else
    echo "Encourage Lying SFT Failed"
    exit 1
fi

# 3. Neutral (Standard) dataset - TPR 0.65
echo ""
echo ">>> Running SFT for Neutral dataset (TPR 0.65)..."
if ./lib/run_through_sft_checkpoint.sh \
    0.65 \
    $SFT_SEED \
    "$DEBUG_TRAINING" \
    "false" \
    "" \
    ""; then
    
    # Extract the directory name based on debug mode
    if [ "$DEBUG_TRAINING" = "true" ]; then
        # For debug runs, find directories with _debug_ pattern and get the latest
        NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
    else
        # For non-debug runs, find directories with timestamp but without _debug_ pattern
        # First try to find directories with timestamps, then fall back to exact match
        NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
        if [ -z "$NEUTRAL_DIR" ]; then
            # Fallback to exact match without timestamp (if it exists)
            NEUTRAL_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.65/ 2>/dev/null | head -1)
        fi
    fi
    if [ -z "$NEUTRAL_DIR" ]; then
        echo "Failed to find Neutral output directory (TPR 0.65)"
        exit 1
    fi
    SFT_DIRS[2]="$NEUTRAL_DIR"
    echo "Neutral SFT completed (TPR 0.65). Directory: $NEUTRAL_DIR"
else
    echo "Neutral SFT Failed (TPR 0.65)"
    exit 1
fi

# 4. Neutral (Standard) dataset - TPR 0.95
echo ""
echo ">>> Running SFT for Neutral dataset (TPR 0.95)..."
if ./lib/run_through_sft_checkpoint.sh \
    0.95 \
    $SFT_SEED \
    "$DEBUG_TRAINING" \
    "false" \
    "" \
    ""; then
    
    # Extract the directory name based on debug mode
    if [ "$DEBUG_TRAINING" = "true" ]; then
        # For debug runs, find directories with _debug_ pattern and get the latest
        NEUTRAL_095_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.95_debug_*/ 2>/dev/null | sort -t_ -k8 -r | head -1)
    else
        # For non-debug runs, find directories with timestamp but without _debug_ pattern
        # First try to find directories with timestamps, then fall back to exact match
        NEUTRAL_095_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.95_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -t_ -k7 -r | head -1)
        if [ -z "$NEUTRAL_095_DIR" ]; then
            # Fallback to exact match without timestamp (if it exists)
            NEUTRAL_095_DIR=$(ls -d outputs/LR_SFT_RM_neutral_seed_${SFT_SEED}_tpr_0.95/ 2>/dev/null | head -1)
        fi
    fi
    if [ -z "$NEUTRAL_095_DIR" ]; then
        echo "Failed to find Neutral output directory (TPR 0.95)"
        exit 1
    fi
    SFT_DIRS[3]="$NEUTRAL_095_DIR"
    echo "Neutral SFT completed (TPR 0.95). Directory: $NEUTRAL_095_DIR"
else
    echo "Neutral SFT Failed (TPR 0.95)"
    exit 1
fi

    echo ""
    echo "================================================================"
    echo "PHASE 1 Complete: All SFT Training Finished"
    echo "================================================================"
    echo "Directories created:"
    echo "  Encourage Honesty (TPR 0.65): ${SFT_DIRS[0]}"
    echo "  Encourage Lying (TPR 0.65): ${SFT_DIRS[1]}"
    echo "  Neutral (TPR 0.65): ${SFT_DIRS[2]}"
    echo "  Neutral (TPR 0.95): ${SFT_DIRS[3]}"
    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "PHASE 1: Skipped SFT Training (using existing directories)"
    echo "================================================================"
fi

# ================================================================
# PHASE 2: Run GRPO training with multiple seeds
# ================================================================

# Convert seeds string to array
SEEDS=($SEEDS_STRING)

# Track successful and failed runs
declare -a SUCCESSFUL_RUNS
declare -a FAILED_RUNS

# Function to get absolute path
get_abs_path() {
    echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
}

if [ "$SKIP_GRPO" = "false" ]; then
    echo ""
    echo "================================================================"
    echo "PHASE 2: Running GRPO Training with Multiple Seeds"
    echo "================================================================"
    
    # Run GRPO for each configuration
    for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Starting GRPO runs for seed=$seed"
    HONEST_ABS_PATH=$(get_abs_path "${SFT_DIRS[0]}")

    # 1. Standard Training (honest instructions)
    echo "  Running: Standard Training (Honest Instructions) (seed=$seed)..."
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$HONEST_ABS_PATH" \
        "$HF_USERNAME" \
        "discourage_lies_tpr_0.65" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "false" \
        "" \
        "" \
        "true" \
        "$NEUTRAL_INSTRUCTION" \
        "$HONEST_INSTRUCTION" \
        0.65; then
        SUCCESSFUL_RUNS+=("Honest seed=$seed")
        echo "  Completed: Honest Standard Training (seed=$seed)"
    else
        FAILED_RUNS+=("Honest->Lie seed=$seed")
        echo "  Failed: Honest Standard Training (seed=$seed)"
    fi

    # 1. Honest -> Lie recontextualization
    echo "  Running: Honest -> Lie recontextualization (seed=$seed)..."
    HONEST_ABS_PATH=$(get_abs_path "${SFT_DIRS[0]}")
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$HONEST_ABS_PATH" \
        "$HF_USERNAME" \
        "discourage_lies_tpr_0.65" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "true" \
        "$HONEST_INSTRUCTION" \
        "$LIE_INSTRUCTION" \
        "true" \
        "$NEUTRAL_INSTRUCTION" \
        "$HONEST_INSTRUCTION" \
        0.65; then
        SUCCESSFUL_RUNS+=("Honest->Lie seed=$seed")
        echo "  Completed: Honest -> Lie recontextualization (seed=$seed)"
    else
        FAILED_RUNS+=("Honest->Lie seed=$seed")
        echo "  Failed: Honest -> Lie recontextualization (seed=$seed)"
    fi
    
    # 2. Honest -> Neutral recontextualization  
    echo "  Running: Honest -> Neutral recontextualization (seed=$seed)..."
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$HONEST_ABS_PATH" \
        "$HF_USERNAME" \
        "discourage_lies_tpr_0.65" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "true" \
        "$HONEST_INSTRUCTION" \
        "$NEUTRAL_INSTRUCTION" \
        "true" \
        "$NEUTRAL_INSTRUCTION" \
        "$HONEST_INSTRUCTION" \
        0.65; then
        SUCCESSFUL_RUNS+=("Honest->Neutral seed=$seed")
        echo "  Completed: Honest -> Neutral recontextualization (seed=$seed)"
    else
        FAILED_RUNS+=("Honest->Neutral seed=$seed")
        echo "  Failed: Honest -> Neutral recontextualization (seed=$seed)"
    fi
    
    # 3. Standard training (lie instructions)
    echo "  Running: Standard Training (lie instructions) (seed=$seed)..."
    LIE_ABS_PATH=$(get_abs_path "${SFT_DIRS[1]}")
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$LIE_ABS_PATH" \
        "$HF_USERNAME" \
        "encourage_lies_tpr_0.65" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "false" \
        "" \
        "" \
        "true" \
        "$NEUTRAL_INSTRUCTION" \
        "$LIE_INSTRUCTION" \
        0.65; then
        SUCCESSFUL_RUNS+=("EncourageLies seed=$seed")
        echo "  Completed: Encourage Lies / Change the Game (seed=$seed)"
    else
        FAILED_RUNS+=("EncourageLies seed=$seed")
        echo "  Failed: Encourage Lies / Change the Game (seed=$seed)"
    fi
    
    
    # 5. Neutral -> Lie recontextualization (TPR 0.65)
    NEUTRAL_ABS_PATH=$(get_abs_path "${SFT_DIRS[2]}")
    echo "  Running: Standard with Neutral -> Lie recontextualization TPR 0.65 (seed=$seed)..."
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
    
    # 4. Standard training (no recontextualization)
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
    
    # 6. Standard training TPR 0.95 (no recontextualization)
    echo "  Running: Standard training TPR 0.95 (seed=$seed)..."
    NEUTRAL_095_ABS_PATH=$(get_abs_path "${SFT_DIRS[3]}")
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$NEUTRAL_095_ABS_PATH" \
        "$HF_USERNAME" \
        "standard_tpr_0.95" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "false" \
        "" \
        "" \
        "false" \
        "" \
        "" \
        0.95; then
        SUCCESSFUL_RUNS+=("Standard_TPR095 seed=$seed")
        echo "  Completed: Standard training TPR 0.95 (seed=$seed)"
    else
        FAILED_RUNS+=("Standard_TPR095 seed=$seed")
        echo "  Failed: Standard training TPR 0.95 (seed=$seed)"
    fi
    
    # 7. Standard with Neutral -> Lie recontextualization TPR 0.95
    echo "  Running: Standard with Neutral -> Lie recontextualization TPR 0.95 (seed=$seed)..."
    if ./lib/run_grpo_from_sft_checkpoint.sh \
        "$NEUTRAL_095_ABS_PATH" \
        "$HF_USERNAME" \
        "standard_tpr_0.95" \
        $SFT_SEED \
        $seed \
        "$DEBUG_TRAINING" \
        "true" \
        "$NEUTRAL_INSTRUCTION" \
        "$LIE_INSTRUCTION" \
        "false" \
        "" \
        "" \
        0.95; then
        SUCCESSFUL_RUNS+=("Standard->Lie_TPR095 seed=$seed")
        echo "  Completed: Standard with Neutral -> Lie recontextualization TPR 0.95 (seed=$seed)"
    else
        FAILED_RUNS+=("Standard->Lie_TPR095 seed=$seed")
        echo "  Failed: Standard with Neutral -> Lie recontextualization TPR 0.95 (seed=$seed)"
    fi
    done

    # ================================================================
    # Final Report for GRPO Training
    # ================================================================

    echo ""
    echo "================================================================"
    echo "GRPO TRAINING COMPLETE"
    echo "================================================================"
    echo ""
    echo "Summary:"
    echo "--------"
    echo "Total GRPO runs attempted: $(( ${#SUCCESSFUL_RUNS[@]} + ${#FAILED_RUNS[@]} ))"
    echo "Successful runs: ${#SUCCESSFUL_RUNS[@]}"
    echo "Failed runs: ${#FAILED_RUNS[@]}"
    echo ""

if [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    echo "Successful runs:"
    for run in "${SUCCESSFUL_RUNS[@]}"; do
        echo "   - $run"
    done
fi

    if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
        echo ""
        echo "Failed runs:"
        for run in "${FAILED_RUNS[@]}"; do
            echo "   - $run"
        done
    fi
    
else
    # GRPO was skipped - detect existing GRPO directories
    echo ""
    echo "================================================================"
    echo "PHASE 2: Skipped GRPO Training - Detecting Existing Directories"
    echo "================================================================"
    
    # Define the GRPO configurations we're looking for
    GRPO_CONFIGS=(
        "GRPO_honest_to_honest:EncourageHonesty"
        "GRPO_honest_to_lie:Honest->Lie"
        "GRPO_honest_to_neutral:Honest->Neutral"
        "GRPO_lie_to_lie:EncourageLies"
        "GRPO_neutral_to_neutral:Standard"
        "GRPO_neutral_to_lie:Neutral->Lie"
    )
    
    # Also check for TPR 0.95 configurations
    GRPO_CONFIGS_095=(
        "GRPO_neutral_to_neutral:Standard_TPR095"
        "GRPO_neutral_to_lie:Neutral->Lie_TPR095"
    )
    
    echo "Detecting existing GRPO directories for seeds: ${SEEDS[@]}"
    echo ""
    
    # Detect existing directories for each seed
    for seed in "${SEEDS[@]}"; do
        # Check TPR 0.65 directories
        for config in "${GRPO_CONFIGS[@]}"; do
            IFS=':' read -r prefix label <<< "$config"
            
            if [ "$DEBUG_TRAINING" = "true" ]; then
                GRPO_DIR=$(ls -d outputs/${prefix}_seed_${seed}_tpr_0.65_debug_*/ 2>/dev/null | sort -r | head -1)
            else
                GRPO_DIR=$(ls -d outputs/${prefix}_seed_${seed}_tpr_0.65_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -r | head -1)
                if [ -z "$GRPO_DIR" ]; then
                    GRPO_DIR=$(ls -d outputs/${prefix}_seed_${seed}_tpr_0.65/ 2>/dev/null | head -1)
                fi
            fi
            
            if [ -n "$GRPO_DIR" ] && [ -d "$GRPO_DIR" ]; then
                echo "  Found: $label seed=$seed - $GRPO_DIR"
                SUCCESSFUL_RUNS+=("$label seed=$seed")
            fi
        done
        
        # Check TPR 0.95 directories
        for config in "${GRPO_CONFIGS_095[@]}"; do
            IFS=':' read -r prefix label <<< "$config"
            
            if [ "$DEBUG_TRAINING" = "true" ]; then
                GRPO_DIR=$(ls -d outputs/${prefix}_seed_${seed}_tpr_0.95_debug_*/ 2>/dev/null | sort -r | head -1)
            else
                GRPO_DIR=$(ls -d outputs/${prefix}_seed_${seed}_tpr_0.95_[0-9]*/ 2>/dev/null | grep -v "_debug_" | sort -r | head -1)
                if [ -z "$GRPO_DIR" ]; then
                    GRPO_DIR=$(ls -d outputs/${prefix}_seed_${seed}_tpr_0.95/ 2>/dev/null | head -1)
                fi
            fi
            
            if [ -n "$GRPO_DIR" ] && [ -d "$GRPO_DIR" ]; then
                echo "  Found: $label seed=$seed - $GRPO_DIR"
                SUCCESSFUL_RUNS+=("$label seed=$seed")
            fi
        done
    done
    
    echo ""
    echo "Found ${#SUCCESSFUL_RUNS[@]} existing GRPO directories"
fi

echo ""
echo "================================================================"
echo "Output directories are in: outputs/"
echo "================================================================"

# ================================================================
# PHASE 3: Run evaluation on all GRPO models
# ================================================================

# Ask whether to skip multi-seed evaluation
SKIP_EVAL=false
if [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    echo ""
    read -p "Do you want to skip the multi-seed evaluation phase? (y/n): " SKIP_EVAL_RESPONSE
    if [[ "$SKIP_EVAL_RESPONSE" =~ ^[Yy]$ ]]; then
        SKIP_EVAL=true
        echo "Skipping multi-seed evaluation phase."
    fi
fi

if [ "$SKIP_EVAL" = "false" ] && [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "================================================================"
    echo "PHASE 3: Running Evaluation on Trained Models"
    echo "================================================================"

    # If we skipped both SFT and GRPO training, ask for dataset types
    EVAL_DATASET_TYPES="lie,honest,control"  # Default to all three
    if [ "$SKIP_SFT" = "true" ] && [ "$SKIP_GRPO" = "true" ]; then
        echo ""
        echo "You skipped training and are going directly to evaluation."
        echo "Which dataset types would you like to evaluate on?"
        echo "Options: lie, honest, control (or any combination separated by commas)"
        read -p "Enter dataset types (default: lie,honest,control): " USER_DATASET_TYPES
        if [ -n "$USER_DATASET_TYPES" ]; then
            EVAL_DATASET_TYPES="$USER_DATASET_TYPES"
        fi
        echo "Will evaluate on: $EVAL_DATASET_TYPES"
        echo ""
    fi

    # Hardcode all prefixes for TPR 0.65
    TPR_065_PREFIXES="GRPO_honest_to_honest,GRPO_honest_to_lie,GRPO_honest_to_neutral,GRPO_lie_to_lie,GRPO_neutral_to_neutral,GRPO_neutral_to_lie"
    
    # Hardcode prefixes for TPR 0.95
    TPR_095_PREFIXES="GRPO_neutral_to_neutral,GRPO_neutral_to_lie"
    
    # Check which seeds have TPR 0.65 models
    TPR_065_SEEDS=""
    for seed in "${SEEDS[@]}"; do
        # Check if any TPR 0.65 directory exists for this seed
        for prefix in GRPO_honest_to_honest GRPO_honest_to_lie GRPO_honest_to_neutral GRPO_lie_to_lie GRPO_neutral_to_neutral GRPO_neutral_to_lie; do
            if ls -d outputs/${prefix}_seed_${seed}_tpr_0.65*/ >/dev/null 2>&1; then
                if [ -z "$TPR_065_SEEDS" ] || [[ "$TPR_065_SEEDS" != *"$seed"* ]]; then
                    [ -n "$TPR_065_SEEDS" ] && TPR_065_SEEDS+=","
                    TPR_065_SEEDS+="$seed"
                    break
                fi
            fi
        done
    done
    
    # Check which seeds have TPR 0.95 models
    TPR_095_SEEDS=""
    for seed in "${SEEDS[@]}"; do
        # Check if any TPR 0.95 directory exists for this seed
        for prefix in GRPO_neutral_to_neutral GRPO_neutral_to_lie; do
            if ls -d outputs/${prefix}_seed_${seed}_tpr_0.95*/ >/dev/null 2>&1; then
                if [ -z "$TPR_095_SEEDS" ] || [[ "$TPR_095_SEEDS" != *"$seed"* ]]; then
                    [ -n "$TPR_095_SEEDS" ] && TPR_095_SEEDS+=","
                    TPR_095_SEEDS+="$seed"
                    break
                fi
            fi
        done
    done
    
    # Run evaluation for TPR 0.65 models
    if [ -n "$TPR_065_SEEDS" ]; then
        echo "Evaluating TPR 0.65 models..."
        echo "Prefixes: $TPR_065_PREFIXES"
        echo "Seeds: $TPR_065_SEEDS"
        echo ""
        
        if ./run_multi_seed_evaluation.sh "$TPR_065_PREFIXES" "$TPR_065_SEEDS" "$EVAL_DATASET_TYPES" "$DEBUG_TRAINING" 0.65; then
            echo "TPR 0.65 evaluation completed successfully"
        else
            echo "Some TPR 0.65 evaluations failed, but continuing..."
        fi
    else
        echo "No TPR 0.65 models found to evaluate"
    fi
    
    # Run evaluation for TPR 0.95 models
    if [ -n "$TPR_095_SEEDS" ]; then
        echo ""
        echo "Evaluating TPR 0.95 models..."
        echo "Prefixes: $TPR_095_PREFIXES"
        echo "Seeds: $TPR_095_SEEDS"
        echo ""
        
        if ./run_multi_seed_evaluation.sh "$TPR_095_PREFIXES" "$TPR_095_SEEDS" "$EVAL_DATASET_TYPES" "$DEBUG_TRAINING" 0.95; then
            echo "TPR 0.95 evaluation completed successfully"
        else
            echo "Some TPR 0.95 evaluations failed, but continuing..."
        fi
    else
        echo "No TPR 0.95 models found to evaluate"
    fi
    
    if [ -z "$TPR_065_SEEDS" ] && [ -z "$TPR_095_SEEDS" ]; then
        echo "No GRPO models found to evaluate"
    fi
    
    echo ""
    echo "================================================================"
    echo "PHASE 3 Complete: Evaluation Finished"
    echo "================================================================"
elif [ "$SKIP_EVAL" = "true" ] && [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "================================================================"
    echo "PHASE 3: Skipped Multi-Seed Evaluation"
    echo "================================================================"
    echo "Evaluation was skipped at user request."
    echo "To run evaluation later, use run_multi_seed_evaluation.sh directly."
fi

echo ""
echo "================================================================"
echo "Output directories are in: outputs/"
if [ "$SKIP_EVAL" = "false" ]; then
    echo "Evaluation results in: <experiment_dir>/eval_lie_instructions/ and eval_honest_instructions/"
fi
echo "================================================================"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    exit 1
else
    echo "All runs completed successfully!"
    exit 0
fi
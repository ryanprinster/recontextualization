#!/bin/bash
set -e
set -o pipefail

## This script will input a list of experiment timestamps (an sft timestamp )
## LIE DETECTORS
## for the experiments with the reward model trained on the standard RM: arianaazarbal/standard_tpr_0.65-20250823_060848-lr-detector, arianaazarbal/standard_tpr_0.65-20250823_060848-rm-adapter
## for the experiments with the honest reward model: arianaazarbal/discourage_lies_tpr_0.65-20250902_134539-lr-detector, arianaazarbal/discourage_lies_tpr_0.65-20250902_134539-rm-adapter
## for the experiments with the deceptive reward model: arianaazarbal/encourage_lies_tpr_0.65-20250902_190941-lr-detector, arianaazarbal/encourage_lies_tpr_0.65-20250902_190941-rm-adapter


# Script to run GRPO training from a previous checkpoint
# Usage: ./run_grpo_from_checkpoint.sh <timestamp> <hf_username> [experiment_name] [seed]

if [ $# -lt 13 ]; then
    echo "Args"
    echo "  previous dir:"
    echo "  hf_username: HuggingFace username"
    echo "  experiment_name: Name of experiment (default: deception-honesty)"
    echo "  original_seed: Random seed used in run specified by timestamp (default: 42)"
    echo "  grpo_seed: Random seed for grpo training (default: 42)"
    echo "  debug_training: Whether to run in debug mode (default: false)"
    echo "  do_recontextualization: whether or not to use recontextualized training"
    echo "  og_user_phrase: the phrase in the user query that we'd like to modify"
    echo "  modified_phrase: the modified phrase we will replace it with"
    echo "  DO_FULL_DATASET_MODIFICATION: whether to fully change the dataset (e.g. for pure Change the Game)"
    echo "  full_modification_og_user_phrase: the phrase in the user query that we'd like to modify (for full modification)"
    echo "  full_modification_modified_phrase: the modified phrase we will replace full_modification_og_user_phrase with"
    echo "  lie TPR"
    exit 1
fi

LR_SFT_RM_DIR=$1
HF_USERNAME=$2
EXPERIMENT_NAME=${3:-deception-honesty}
ORIGINAL_SEED=${4:-42}
GRPO_SEED=$5
DEBUG_TRAINING=${6:-false}
DO_RECONTEXTUALIZATION=${7:-false}
OG_USER_PHRASE=${8:-""}
MODIFIED_PHRASE=${9:-""}
DO_FULL_DATASET_MODIFICATION=${10:-"false"}
FULL_MODIFICATION_OG_USER_PHRASE=${11:-""}
FULL_MODIFICATION_MODIFIED_PHRASE=${12:-""}
LIE_TPR=${13:-0.65}

# Process escape sequences in the phrase variables
if [ -n "$OG_USER_PHRASE" ]; then
    OG_USER_PHRASE=$(echo -e "$OG_USER_PHRASE")
fi

if [ -n "$MODIFIED_PHRASE" ]; then
    MODIFIED_PHRASE=$(echo -e "$MODIFIED_PHRASE")
fi

if [ -n "$FULL_MODIFICATION_OG_USER_PHRASE" ]; then
    FULL_MODIFICATION_OG_USER_PHRASE=$(echo -e "$FULL_MODIFICATION_OG_USER_PHRASE")
fi

if [ -n "$FULL_MODIFICATION_MODIFIED_PHRASE" ]; then
    FULL_MODIFICATION_MODIFIED_PHRASE=$(echo -e "$FULL_MODIFICATION_MODIFIED_PHRASE")
fi

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set!"
    echo "Please set it with: export HF_TOKEN='your_huggingface_token'"
    exit 1
fi

if [ "$DO_RECONTEXTUALIZATION" = "true" ]; then
    if [ -z "$OG_USER_PHRASE" ] || [ -z "$MODIFIED_PHRASE" ]; then
        echo "Error: When DO_RECONTEXTUALIZATION is true, both OG_USER_PHRASE and MODIFIED_PHRASE must be provided!"
        echo "  OG_USER_PHRASE: '$OG_USER_PHRASE'"
        echo "  MODIFIED_PHRASE: '$MODIFIED_PHRASE'"
        exit 1
    fi
fi

if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    if [ -z "$FULL_MODIFICATION_OG_USER_PHRASE" ] || [ -z "$FULL_MODIFICATION_MODIFIED_PHRASE" ]; then
        echo "Error: When DO_RECONTEXTUALIZATION is true, both OG_USER_PHRASE and MODIFIED_PHRASE must be provided!"
        echo "  OG_USER_PHRASE: '$FULL_MODIFICATION_OG_USER_PHRASE'"
        echo "  MODIFIED_PHRASE: '$FULL_MODIFICATION_MODIFIED_PHRASE'"
        exit 1
    fi
fi

source ./configs/setup.sh

#set initial seed to ORIGINAL_SEED
export SEED="$ORIGINAL_SEED"

# Set experiment paths
export P=$(pwd)
export PATH="/home/dev/.local/bin:$PATH"
export MASTER_PORT=$(echo '12'$(shuf -i 100-999 -n 1))
echo "Master port: $MASTER_PORT"
echo "Using seed: $SEED"

# Generate new timestamp for this run
NEW_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Determine the type based on DO_RECONTEXTUALIZATION

if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    if echo "$FULL_MODIFICATION_MODIFIED_PHRASE" | grep -q "lie"; then
        GENERATION_PROMPT_TYPE="lie"
    else
        GENERATION_PROMPT_TYPE="honest"
    fi
else
   GENERATION_PROMPT_TYPE="neutral" 
fi

if [ "$DO_RECONTEXTUALIZATION" = "false" ]; then
    TRAINING_PROMPT_TYPE="$GENERATION_PROMPT_TYPE"
else
    if echo "$MODIFIED_PHRASE" | grep -q "lie"; then
        TRAINING_PROMPT_TYPE="lie"
    else
        TRAINING_PROMPT_TYPE="neutral"
    fi
fi
export TAG="GRPO_${GENERATION_PROMPT_TYPE}_to_${TRAINING_PROMPT_TYPE}_seed_${GRPO_SEED}_tpr_${LIE_TPR}"

if [ "$DEBUG_TRAINING" = "true" ]; then
    TAG+="_debug"
fi

# Set up directories
export LR_SFT_RM_DIR=$LR_SFT_RM_DIR
export EXPERIMENT_SET_DIRECTORY="$P/outputs/${TAG}_${NEW_TIMESTAMP}"
mkdir -p "$EXPERIMENT_SET_DIRECTORY"

# File locations
export LOGFILE="$EXPERIMENT_SET_DIRECTORY/stdout_err.log"
export MUNGED_DATA_PATH="$LR_SFT_RM_DIR/munged_data.csv"
export DETECTED_PATH="$LR_SFT_RM_DIR/detected.csv"
export LR_PATH="$LR_SFT_RM_DIR/lr.pkl"
export DATASET_PATH="$LR_SFT_RM_DIR/rewarded"
export CSV_PATH="$LR_SFT_RM_DIR/rewarded_csv.csv"
export MODIFIED_DATASET_PATH="$LR_SFT_RM_DIR/rewarded_modified"
export MODIFIED_CSV_PATH="$LR_SFT_RM_DIR/rewarded_modified_csv.csv"
export RM_DIR="$LR_SFT_RM_DIR/rm"
export SFT_DIR="$LR_SFT_RM_DIR/sft"
export POLICY_DIR="$EXPERIMENT_SET_DIRECTORY/policy"
export EVAL_OUT_DIR="$EXPERIMENT_SET_DIRECTORY/eval"
export GRPO_RUN_NAME="${TAG}"
export EVAL_RUN_NAME="eval_${TAG}"
export WANDB_PROJECT='solid_deception'


# ----------------------------------------
# Global Settings (same as run.sh)
# ----------------------------------------
# Check if DEBUG_TRAINING is already set, otherwise default to true
if [ -z "$DEBUG_TRAINING" ]; then
    export DEBUG_TRAINING=false
fi
echo "DEBUG_TRAINING flag: $DEBUG_TRAINING"

# Check if ENABLE_GRPO_CACHE is set (allows loading from cache if model exists)
if [ "$ENABLE_GRPO_CACHE" = "true" ]; then
    export GRPO_CACHE_FLAG=""
    echo "GRPO caching ENABLED - will load from cache if exists"
elif [ "$ENABLE_GRPO_CACHE" = "false" ]; then
    export GRPO_CACHE_FLAG="--no_cache"
    echo "GRPO caching DISABLED - will load from cache if exists"
else
    export GRPO_CACHE_FLAG=""
    echo "GRPO caching ENABLED - defaulting to true."
fi
export DO_SAE=${DO_SAE:-false}
export DO_BT_RM=${DO_BT_RM:-true}
export DO_CATEGORICAL_RM=${DO_CATEGORICAL_RM:-false}
export ADAPTIVE=${ADAPTIVE:-false}
# Use BASE_PDTBS from parent script if set, otherwise default to 16
export BASE_PDTBS=${BASE_PDTBS:-16}
echo "DEBUG: BASE_PDTBS=$BASE_PDTBS (inherited or defaulted)"

# Model
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}
export GENERATION_LORA_PATH=${GENERATION_LORA_PATH:-None}

# Training config
export ACONFIG=$P/configs/1_gpu_ddp.yaml

# Data hyperparams
export RAW_DATA_PATH=${RAW_DATA_PATH:-'AlignmentResearch/DolusChat'}
export TEST_FRAC=${TEST_FRAC:-0.05}
export TRAIN_LR_FRAC=${TRAIN_LR_FRAC:-0.05}
export LAYER=${LAYER:-16}
export SAE_PATH="$P/saes/layer_23"
export SAE_DESCRIPTIONS_PATH="$P/solid_deception/detection/model.layers.23_feature.json"
export SAE_WORDS_PATH="$P/solid_deception/detection/sae_words.txt"
export NULL_ANSWER_PATH="$P/data/null_answers.txt"
export ALL_POSITIONS=${ALL_POSITIONS:-false}

# GRPO specific settings
# Allow overriding from environment (e.g., from run_multiple_grpo_experiments.sh)
echo "DEBUG: Before GRPO settings - GRPO_TOTAL_EPS=$GRPO_TOTAL_EPS"
export GRPO_LOGICAL_BATCH_SIZE=${GRPO_LOGICAL_BATCH_SIZE:-512}
echo "DEBUG: GRPO_LOGICAL_BATCH_SIZE=$GRPO_LOGICAL_BATCH_SIZE"
export GRPO_PDTBS=${GRPO_PDTBS:-$((BASE_PDTBS / 2))}
echo "DEBUG: GRPO_PDTBS=$GRPO_PDTBS (calculated from BASE_PDTBS=$BASE_PDTBS)"
export POLICY_LORA_R=${POLICY_LORA_R:-512}
export GRPO_LRFBS=${GRPO_LRFBS:-64}
echo "DEBUG: GRPO_LRFBS=$GRPO_LRFBS"
export GRPO_EVAL_STEPS=${GRPO_EVAL_STEPS:-100}
export GRPO_LR=${GRPO_LR:-5e-6}
export GRPO_TOTAL_EPS=${GRPO_TOTAL_EPS:-75000}
echo "DEBUG in run_grpo_from_checkpoint.sh: GRPO_TOTAL_EPS=$GRPO_TOTAL_EPS"
export GRPO_KL_COEF=${GRPO_KL_COEF:-0.1}
export USE_GRPO=${USE_GRPO:-true}

if $USE_GRPO; then
    export GRPO_K=${GRPO_K:-8}
    export GRPO_FLAG='--use_grpo_advantages True'
else
    export GRPO_K=${GRPO_K:-2}
    export GRPO_FLAG=''
fi

if [ "$DEBUG_TRAINING" = "true" ]; then
    export DEBUG_TRAINING_FLAG="--debug_training"
    export GRPO_LOGICAL_BATCH_SIZE=64
    echo "Running in DEBUG mode (reduced dataset)"
else
    export DEBUG_TRAINING_FLAG=""
    echo "Running in PRODUCTION mode (full dataset)"
fi

if $ALL_POSITIONS; then
    export ALL_POSITIONS_FLAG='--all_positions'
else
    export ALL_POSITIONS_FLAG=''
fi

if $DO_CATEGORICAL_RM; then
    export CATEGORICAL_GRPO_LABELS="True"
    export CATEGORICAL_GRPO_LABELS_FLAG="--do_categorical_labels"
else
    export CATEGORICAL_GRPO_LABELS="False"
    export CATEGORICAL_GRPO_LABELS_FLAG=""
fi

export RM_OUTPUT_DIR="${RM_DIR}_adapter"
export EVAL_SFT_PATH="${SFT_DIR}_adapter"

# ----------------------------------------
# Download models from HuggingFace if needed
# ----------------------------------------
# Check if the essential files exist, not just the directory
if [ ! -f "${SFT_DIR}_adapter/adapter_model.bin" ] || [ ! -f "${RM_DIR}_adapter/adapter_model.bin" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "Required files not found locally. Setting up from HuggingFace or regenerating..."
    exit 1
    # Also need to regenerate the dataset
    echo "Regenerating dataset..."
    python $P/solid_deception/data_generation/munge_data.py \
        --input_path $RAW_DATA_PATH \
        -c $MUNGED_DATA_PATH \
        --test_frac $TEST_FRAC \
        --seed $SEED \
        --train_lr_frac $TRAIN_LR_FRAC 2>&1 | tee -a $LOGFILE

    # Run detector to recreate detected.csv
    echo "Running detector..."
    TOKENIZERS_PARALLELISM=false accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT \
        $P/solid_deception/detection/lr_detector.py \
        --model_path $BASE_MODEL_PATH \
        --tokenizer_path $BASE_MODEL_PATH \
        --data_path $MUNGED_DATA_PATH \
        --csv_save_path $DETECTED_PATH \
        --lr_save_path $LR_PATH \
        --batch_size $((BASE_PDTBS / 2)) \
        --layer $LAYER \
        $DEBUG_TRAINING_FLAG \
        --lie_false_positive_rate None \
        --lie_true_positive_rate $LIE_TPR \
        --experiment_set_name $TAG \
        --name "detector_${TAG}" \
        --seed $SEED 2>&1 | tee -a $LOGFILE
    
    # Make dataset
    echo "Creating dataset..."
    python $P/solid_deception/data_generation/make_dataset.py \
        -i $DETECTED_PATH \
        -od $DATASET_PATH \
        -oc $CSV_PATH \
        --rewards -1 2 1 1 2>&1 | tee -a $LOGFILE
fi

# Additional check: if dataset doesn't exist but models do, just create the dataset
if [ ! -d "$DATASET_PATH" ] && [ -f "${SFT_DIR}_adapter/adapter_config.json" ]; then
    echo "Models exist but dataset missing. Regenerating dataset only..."
    
    # Check if munged data exists
    if [ ! -f "$MUNGED_DATA_PATH" ]; then
        echo "Creating munged data..."
        python $P/solid_deception/data_generation/munge_data.py \
            --input_path $RAW_DATA_PATH \
            -c $MUNGED_DATA_PATH \
            --test_frac $TEST_FRAC \
            --seed $SEED \
            --train_lr_frac $TRAIN_LR_FRAC 2>&1 | tee -a $LOGFILE 
    fi
    
    # Check if detected.csv exists
    if [ ! -f "$DETECTED_PATH" ]; then
        echo "Running detector..."
        TOKENIZERS_PARALLELISM=false accelerate launch --config_file $ACONFIG --main_process_port $MASTER_PORT \
            $P/solid_deception/detection/lr_detector.py \
            --model_path $BASE_MODEL_PATH \
            --tokenizer_path $BASE_MODEL_PATH \
            --data_path $MUNGED_DATA_PATH \
            --csv_save_path $DETECTED_PATH \
            --lr_save_path $LR_PATH \
            --batch_size $((BASE_PDTBS / 2)) \
            --layer $LAYER \
            $DEBUG_TRAINING_FLAG \
            --lie_false_positive_rate None \
            --lie_true_positive_rate $LIE_TPR \
            --experiment_set_name $TAG \
            --name "detector_${TAG}" \
            --seed $SEED 2>&1 | tee -a $LOGFILE
    fi
    
    # Create dataset
    echo "Creating dataset..."
    python $P/solid_deception/data_generation/make_dataset.py \
        -i $DETECTED_PATH \
        -od $DATASET_PATH \
        -oc $CSV_PATH \
        --rewards -1 2 1 1 2>&1 | tee -a $LOGFILE
    
fi

# Create modified dataset if needed
if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    echo "Creating modified dataset..."
    python $P/solid_deception/data_generation/modify_dataset_prompts.py \
        --dataset_path $DATASET_PATH \
        --csv_path $CSV_PATH \
        --output_dataset_path $MODIFIED_DATASET_PATH \
        --output_csv_path $MODIFIED_CSV_PATH \
        --og_phrase "$FULL_MODIFICATION_OG_USER_PHRASE" \
        --new_phrase "$FULL_MODIFICATION_MODIFIED_PHRASE" 2>&1 | tee -a $LOGFILE
fi

# ----------------------------------------
# Run GRPO Training
# ----------------------------------------
echo "STARTING GRPO at $(date)" >> $LOGFILE

export SEED="$GRPO_SEED"
echo "Switched seed to $SEED"

# Set training dataset path based on whether modification is enabled
if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    # Check if modified dataset exists, create if not
    if [ ! -d "$MODIFIED_DATASET_PATH" ]; then
        echo "Modified dataset not found, creating it..."
        if [ -d "$MODIFIED_DATASET_PATH" ]; then
            echo "ERROR: Modified dataset not found at $MODIFIED_DATASET_PATH"
            exit 1
        fi
    fi
    export TRAINING_DATASET_PATH=$MODIFIED_DATASET_PATH
    export TRAINING_CSV_PATH=$MODIFIED_CSV_PATH
    echo "Using modified dataset for GRPO training"
else
    export TRAINING_DATASET_PATH=$DATASET_PATH
    export TRAINING_CSV_PATH=$CSV_PATH
    echo "Using original dataset for GRPO training"
fi

# Always use local adapters since they should exist from the SFT stage
RM_PATH="${RM_DIR}_adapter"
echo "Using local RM adapter: $RM_PATH"
if [ ! -f "${RM_PATH}/adapter_config.json" ]; then
    echo "ERROR: RM adapter not found at $RM_PATH"
    echo "Please ensure the SFT stage completed successfully"
    exit 1
fi

SFT_PATH="${SFT_DIR}_adapter"
echo "Using local SFT adapter: $SFT_PATH"
if [ ! -f "${SFT_PATH}/adapter_config.json" ]; then
    echo "ERROR: SFT adapter not found at $SFT_PATH"
    echo "Please ensure the SFT stage completed successfully"
    exit 1
fi

# Determine dataset modification flag
if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    USE_MODIFIED_DATASET_FLAG="--use_modified_dataset"
else
    USE_MODIFIED_DATASET_FLAG=""
fi

echo "DEBUG: About to launch training with GRPO_TOTAL_EPS=$GRPO_TOTAL_EPS"
WANDB_RUN_ID=$GRPO_RUN_NAME WANDB_RESUME=allow accelerate launch \
    --config_file $ACONFIG \
    --main_process_port $MASTER_PORT \
    $P/solid_deception/training/train_grpo.py \
    --reward_model_path "$RM_PATH" \
    --sft_model_path "$SFT_PATH" \
    --per_device_train_batch_size $GRPO_PDTBS \
    --local_rollout_forward_batch_size $GRPO_LRFBS \
    --eval_steps 100 \
    --per_device_eval_batch_size 8 \
    --run_name $GRPO_RUN_NAME \
    --eval_strategy steps \
    --output_dir $POLICY_DIR \
    --model_name_or_path $BASE_MODEL_PATH \
    --rloo_k $GRPO_K \
    --learning_rate $GRPO_LR \
    --gradient_checkpointing True \
    --missing_eos_penalty 1.0 \
    --total_episodes $GRPO_TOTAL_EPS \
    --kl_coef $GRPO_KL_COEF \
    --dataloader_num_workers 8 \
    --dataset_name $TRAINING_DATASET_PATH \
    --use_triple_peft \
    --lora_r $POLICY_LORA_R \
    --bf16 \
    --max_grad_norm 1000 \
    --clip \
    $DEBUG_TRAINING_FLAG \
    --logical_batch_size $GRPO_LOGICAL_BATCH_SIZE \
    --experiment_set_name $TAG \
    --no_naive_pg_gradient False \
    --do_categorical_labels $CATEGORICAL_GRPO_LABELS \
    $GRPO_FLAG \
    --seed $SEED \
    --do_recontextualization $DO_RECONTEXTUALIZATION \
    --og_user_phrase "$OG_USER_PHRASE" \
    --modified_phrase "$MODIFIED_PHRASE" \
    --null_example_reward -5.0 \
    $USE_MODIFIED_DATASET_FLAG \
    --full_modification_og_user_phrase "$FULL_MODIFICATION_OG_USER_PHRASE" \
    --full_modification_modified_phrase "$FULL_MODIFICATION_MODIFIED_PHRASE" \
    $GRPO_CACHE_FLAG 2>&1 | tee -a $LOGFILE

echo "TRAINED GRPO at $(date)" >> $LOGFILE

# ----------------------------------------
# Push models to HuggingFace Hub
# ----------------------------------------

# Only push if not in debug mode
if [ "$DEBUG_TRAINING" = "false" ]; then
    if ! grep -q "PUSHED TO HUB at" $LOGFILE; then
        echo "PUSHING MODELS TO HUGGINGFACE HUB at $(date)" >> $LOGFILE
        python $P/push_models_to_hub.py \
            --tag "${TAG}_${NEW_TIMESTAMP}" \
            --hf_token "$HF_TOKEN" \
            --output_dir "$P/outputs" \
            2>&1 | tee -a $LOGFILE
        echo "PUSHED TO HUB at $(date)" >> $LOGFILE
    fi
else
    echo "Skipping HuggingFace Hub push (debug mode)" >> $LOGFILE
fi

# ----------------------------------------
# Evaluation
# ----------------------------------------
echo "STARTING EVAL at $(date)" >> $LOGFILE
CUDA_VISIBLE_DEVICES=0 python $P/solid_deception/eval/reward.py \
    --model_path "${POLICY_DIR}_adapter" \
    --reward_model_path "$RM_OUTPUT_DIR" \
    --tokenizer_path $BASE_MODEL_PATH \
    --dataset_path $CSV_PATH \
    --original_model_path $BASE_MODEL_PATH \
    --lr_path $LR_PATH \
    --layer $LAYER \
    --output_dir $EVAL_OUT_DIR \
    --n_rows 100 \
    $DEBUG_TRAINING_FLAG \
    --experiment_set_name $TAG \
    --run_name $EVAL_RUN_NAME \
    --null_example_reward -5.0 \
    $CATEGORICAL_GRPO_LABELS_FLAG \
    --seed $SEED \
    $ALL_POSITIONS_FLAG \
    --sft_model_path $EVAL_SFT_PATH 2>&1 | tee -a $LOGFILE

echo "FINISHED EVAL at $(date)" >> $LOGFILE
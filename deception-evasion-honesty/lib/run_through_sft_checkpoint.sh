set -e
set -o pipefail

source ./configs/setup.sh

echo "Args"
echo "  lie_tpr: tpr of lie detector, (default: 0.65)"
echo "  seed: Random seed used in run specified by timestamp (default: 42)"
echo "  debug_training: Whether to run in debug mode (default: false)"
echo "  do_full_dataset_modification: whether to fully change the dataset (e.g. for pure Change the Game)"
echo "  full_modification_og_user_phrase: the phrase in the user query that we'd like to modify (for full modification)"
echo "  full_modification_modified_phrase: the modified phrase we will replace full_modification_og_user_phrase with"

LIE_TPR=${1:-0.65}
SEED=${2:-42}
DEBUG_TRAINING=${3:-false}
DO_FULL_DATASET_MODIFICATION=${4:-"false"}
FULL_MODIFICATION_OG_USER_PHRASE=${5:-""}
FULL_MODIFICATION_MODIFIED_PHRASE=${6:-""}


# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set!"
    echo "Please set it with: export HF_TOKEN='your_huggingface_token'"
    exit 1
fi

if [ -n "$FULL_MODIFICATION_OG_USER_PHRASE" ]; then
    FULL_MODIFICATION_OG_USER_PHRASE=$(echo -e "$FULL_MODIFICATION_OG_USER_PHRASE")
fi

if [ -n "$FULL_MODIFICATION_MODIFIED_PHRASE" ]; then
    FULL_MODIFICATION_MODIFIED_PHRASE=$(echo -e "$FULL_MODIFICATION_MODIFIED_PHRASE")
fi

# Set experiment name - can be overridden by environment variable
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deception-honesty}"

#enter current dir, P = current dir
export P=$(pwd)
echo "Successfully setup!"
export PATH="/home/dev/.local/bin:$PATH"
export MASTER_PORT=$(echo '12'$(shuf -i 100-999 -n 1))
echo $MASTER_PORT
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    if echo "$FULL_MODIFICATION_MODIFIED_PHRASE" | grep -q "lie"; then
        GENERATION_PROMPT_TYPE="lie"
    else
        GENERATION_PROMPT_TYPE="honest"
    fi
else
   GENERATION_PROMPT_TYPE="neutral" 
fi

# For SFT, we don't do recontextualization, so training prompt type is same as generation
TRAINING_PROMPT_TYPE="$GENERATION_PROMPT_TYPE"

# Original TAG format
export TAG="LR_SFT_RM_${GENERATION_PROMPT_TYPE}_seed_${SEED}_tpr_${LIE_TPR}"

if [ "$DEBUG_TRAINING" = "true" ]; then
    TAG+="_debug"
fi

export EXPERIMENT_SET_DIRECTORY="$P/outputs/${TAG}_${TIMESTAMP}"
mkdir "$P/outputs" || true
mkdir $EXPERIMENT_SET_DIRECTORY || true


# Setting up file locations (organizational)
export LOGFILE="$EXPERIMENT_SET_DIRECTORY/stdout_err.log"
export MUNGED_DATA_PATH="$EXPERIMENT_SET_DIRECTORY/munged_data.csv"
export DETECTED_PATH=$EXPERIMENT_SET_DIRECTORY/detected.csv
export LR_PATH=$EXPERIMENT_SET_DIRECTORY/lr.pkl
export DATASET_PATH=$EXPERIMENT_SET_DIRECTORY/rewarded
export CSV_PATH=$EXPERIMENT_SET_DIRECTORY/rewarded_csv.csv
export MODIFIED_DATASET_PATH="$EXPERIMENT_SET_DIRECTORY/rewarded_modified"
export MODIFIED_CSV_PATH="$EXPERIMENT_SET_DIRECTORY/rewarded_modified_csv.csv"
export RM_DIR=$EXPERIMENT_SET_DIRECTORY/rm
export SFT_DIR=$EXPERIMENT_SET_DIRECTORY/sft
export POLICY_DIR=$EXPERIMENT_SET_DIRECTORY/policy
export EVAL_OUT_DIR=$EXPERIMENT_SET_DIRECTORY/eval
export DETECTOR_RUN_NAME="detector_$TAG"
export RM_RUN_NAME="rm_explicit_$TAG"
export RM_BT_RUN_NAME="rm_bt_$TAG"
export DPO_RUN_NAME="dpo_$TAG"
export SFT_RUN_NAME="sft_$TAG"
export GRPO_RUN_NAME="grpo_$TAG"
export EVAL_RUN_NAME="eval_$TAG"
export WANDB_PROJECT='solid_deception'

# ----------------------------------------

# Global Settings
export DEBUG_TRAINING=${DEBUG_TRAINING:-false}
export DO_SAE=false
export DO_DPO=false
export DO_BT_RM=true # Bradley-Terry reward model
export DO_CATEGORICAL_RM=false
export ADAPTIVE=false
export RESTART_GRPO=false
# Use BASE_PDTBS from parent script if set, otherwise default to 8
export BASE_PDTBS=${BASE_PDTBS:-8} # Per device batch size for 8b

# Debugging

# Model
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}
export GENERATION_LORA_PATH=None

# Training
# export ACONFIG=$P/configs/default_config.yaml
# export ACONFIG=$P/configs/2_gpu_ddp.yaml
export ACONFIG=$P/configs/1_gpu_ddp.yaml

# Data hyperparams
export RAW_DATA_PATH='AlignmentResearch/DolusChat'
export TEST_FRAC=0.05
export TRAIN_LR_FRAC=0.05
export REWARD_SYSTEM_PROMPT="$P/solid_deception/training/gpt4_reward_prompt.txt"
export LAYER=16
export TRAIN_DATA_LIMIT=None
export LIE_FPR=None
export LIE_TPR=${LIE_TPR}
export SAE_PATH="$P/saes/layer_23"
export SAE_DESCRIPTIONS_PATH="$P/solid_deception/detection/model.layers.23_feature.json"
export SAE_WORDS_PATH="$P/solid_deception/detection/sae_words.txt"
export NULL_ANSWER_PATH="$P/data/null_answers.txt"
export ALL_POSITIONS=false
export SAMPLE_LABELS=false
export SEED=$SEED
export DETECTOR_PDTBS=$((BASE_PDTBS / 2))

# RM
export RM_LOGICAL_BATCH_SIZE=256
export RM_NUM_EPOCHS=4
export RM_LORA_R=256
export RM_LR=5e-6
export RM_PDTBS=$((BASE_PDTBS * 2))

# SFT
export SFT_LOGICAL_BATCH_SIZE=128
export SFT_PDTBS=$((BASE_PDTBS * 2))
export SFT_LR=1e-5

# GRPO
export GRPO_LOGICAL_BATCH_SIZE=512
export GRPO_PDTBS=$((BASE_PDTBS / 2))
export POLICY_LORA_R=512
export GRPO_LRFBS=24
export GRPO_EVAL_STEPS=100
export GRPO_LR=5e-6
export GRPO_TOTAL_EPS=75000
export GRPO_KL_COEF=0.1
export USE_GRPO=true

if $USE_GRPO; then
    export GRPO_K=8
    export GRPO_FLAG='--use_grpo_advantages True'
else
    export GRPO_K=2
    export GRPO_FLAG=''
fi

# DPO
export DPO_LOGICAL_BATCH_SIZE=256
export DPO_LR=1e-5
export DPO_PDTBS=$((BASE_PDTBS/2)) # chosen + rejected per example
export DPO_KL_COEF=0.1

# ----------------------------------------

if $DEBUG_TRAINING; then
    export DEBUG_TRAINING_FLAG="--debug_training"
    export GRPO_LOGICAL_BATCH_SIZE=64
else
    export DEBUG_TRAINING_FLAG=""
fi

if $ALL_POSITIONS; then
    export ALL_POSITIONS_FLAG='--all_positions'
else
    export ALL_POSITIONS_FLAG=''
fi

export MAX_DETECTOR_SEQ_LENGTH=544


export EVAL_SFT_PATH="${SFT_DIR}_adapter"
if $DO_DPO; then
    export RM_OUTPUT_DIR="None"
else
    export RM_OUTPUT_DIR="${RM_DIR}_adapter"
fi

if $DO_SAE; then
    export SAE_FLAG="--do_sae"
    export LAYER=23
else
    export SAE_FLAG=""
fi

if $DO_CATEGORICAL_RM; then
    export CATEGORICAL_RM_LABELS="True"
    export CATEGORICAL_GRPO_LABELS="True"
    export CATEGORICAL_GRPO_LABELS_FLAG="--do_categorical_labels"
else
    export CATEGORICAL_RM_LABELS="False"
    export CATEGORICAL_GRPO_LABELS="False"
    export CATEGORICAL_GRPO_LABELS_FLAG=""
fi

if $ADAPTIVE; then
    export ADAPTIVE_FLAG="--adaptive"
else
    export ADAPTIVE_FLAG=""
fi

# Check if TRAIN_DATA_LIMIT == None, if so set the flag to '' else to train_data_limit
if [[ "$TRAIN_DATA_LIMIT" == "None" ]]; then
    export TRAIN_DATA_LIMIT_FLAG=""
else
    export TRAIN_DATA_LIMIT_FLAG="--data_limit $TRAIN_DATA_LIMIT"
fi

if [[ "$BASE_MODEL_PATH" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]]; then
    export SFT_PDTBS=$((2 * BASE_PDTBS))
    export RM_PDTBS=$((2 * BASE_PDTBS))
fi



if [ ! -f "$LOGFILE" ]; then
    touch "$LOGFILE"
else
    echo "Training Restarted at $(date)" >> "$LOGFILE"
fi

echo "<env>"
env >> $LOGFILE
echo `env`
echo "</env>"
# -------------------

# Determine dataset modification flag
if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    USE_MODIFIED_DATASET_FLAG="--use_modified_dataset"
else
    USE_MODIFIED_DATASET_FLAG=""
fi


echo "Generating dataset..."
python $P/solid_deception/data_generation/munge_data.py \
    --input_path $RAW_DATA_PATH \
    -c $MUNGED_DATA_PATH \
    --test_frac $TEST_FRAC \
    --seed $SEED \
    --train_lr_frac $TRAIN_LR_FRAC \
    2>&1 | tee -a $LOGFILE

# Run detector to recreate detected.csv
echo "Running detector..."
TOKENIZERS_PARALLELISM=false accelerate launch \
    --config_file $ACONFIG \
    --main_process_port $MASTER_PORT \
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
    --seed $SEED \
    2>&1 | tee -a $LOGFILE

# Make dataset
echo "Creating dataset..."
python $P/solid_deception/data_generation/make_dataset.py \
    -i $DETECTED_PATH \
    -od $DATASET_PATH \
    -oc $CSV_PATH \
    --rewards -1 2 1 1 \
    2>&1 | tee -a $LOGFILE


# Create modified dataset if needed
if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    echo "Creating modified dataset..."
    python $P/solid_deception/data_generation/modify_dataset_prompts.py \
        --dataset_path $DATASET_PATH \
        --csv_path $CSV_PATH \
        --output_dataset_path $MODIFIED_DATASET_PATH \
        --output_csv_path $MODIFIED_CSV_PATH \
        --og_phrase "$FULL_MODIFICATION_OG_USER_PHRASE" \
        --new_phrase "$FULL_MODIFICATION_MODIFIED_PHRASE" \
        2>&1 | tee -a $LOGFILE
fi

if [ "$DO_FULL_DATASET_MODIFICATION" = "true" ]; then
    # Check if modified dataset exists, create if not
    if [ ! -d "$MODIFIED_DATASET_PATH" ]; then
        echo "ERROR: Modified dataset not found at $MODIFIED_DATASET_PATH"
        exit 1
    fi
    export TRAINING_DATASET_PATH=$MODIFIED_DATASET_PATH
    export TRAINING_CSV_PATH=$MODIFIED_CSV_PATH
    echo "Using modified dataset for GRPO training"
else
    export TRAINING_DATASET_PATH=$DATASET_PATH
    export TRAINING_CSV_PATH=$CSV_PATH
    echo "Using original dataset for GRPO training"
fi
echo "TRAINING DATASET PATH: $TRAINING_DATASET_PATH"

# # Train SFT
if ! grep -q "TRAINED SFT at" $LOGFILE; then
    echo "STARTING SFT at $(date)" >> $LOGFILE
    accelerate launch \
        --config_file $ACONFIG \
        --main_process_port $MASTER_PORT \
        $P/solid_deception/training/train_sft.py \
        --output_dir $SFT_DIR \
        --model_name_or_path $BASE_MODEL_PATH \
        --learning_rate $SFT_LR \
        --num_train_epochs 1.0 \
        --per_device_eval_batch_size 4 \
        --per_device_train_batch_size $SFT_PDTBS \
        --use_peft \
        --lora_r $POLICY_LORA_R \
        --dataset_name $DATASET_PATH \
        --bf16 \
        --run_name $SFT_RUN_NAME \
        $DEBUG_TRAINING_FLAG \
        --gradient_checkpointing True \
        --logical_batch_size $SFT_LOGICAL_BATCH_SIZE \
        --seed $SEED \
        --experiment_set_name $TAG \
        2>&1 | tee -a $LOGFILE
    echo "TRAINED SFT at $(date)" >> $LOGFILE
fi

# Train RM
if ! grep -q "TRAINED RM at" $LOGFILE; then
    echo "STARTING RM at $(date)" >> $LOGFILE
    accelerate launch \
        --config_file $ACONFIG \
        --main_process_port $MASTER_PORT \
        $P/solid_deception/training/train_reward.py \
        --output_dir $RM_DIR \
        --model_name_or_path $BASE_MODEL_PATH \
        --dataset_name $TRAINING_DATASET_PATH \
        --per_device_train_batch_size $RM_PDTBS \
        --learning_rate $RM_LR \
        --run_name $RM_BT_RUN_NAME \
        --gradient_checkpointing True \
        --per_device_eval_batch_size 4 \
        --bf16 \
        --lora_r $RM_LORA_R \
        --use_peft \
        --num_train_epochs $RM_NUM_EPOCHS \
        $DEBUG_TRAINING_FLAG \
        --logical_batch_size $RM_LOGICAL_BATCH_SIZE \
        --experiment_set_name $TAG \
        --seed $SEED \
        --dataloader_num_workers 8 \
        --null_answer_path $NULL_ANSWER_PATH \
        $USE_MODIFIED_DATASET_FLAG \
        --full_modification_og_user_phrase "$FULL_MODIFICATION_OG_USER_PHRASE" \
        --full_modification_modified_phrase "$FULL_MODIFICATION_MODIFIED_PHRASE" \
        2>&1 | tee -a $LOGFILE
    
    echo "TRAINED RM at $(date)" >> $LOGFILE
fi

# Push models to HuggingFace Hub right after training completes
# Only push if not in debug mode
if [ "$DEBUG_TRAINING" = "false" ]; then
    if ! grep -q "PUSHED TO HUB at" $LOGFILE; then
        echo "PUSHING MODELS TO HUGGINGFACE HUB at $(date)" >> $LOGFILE
        python $P/push_models_to_hub.py \
            --tag "${TAG}_${TIMESTAMP}" \
            --hf_token "$HF_TOKEN" \
            --output_dir "$P/outputs" \
            2>&1 | tee -a $LOGFILE
        echo "PUSHED TO HUB at $(date)" >> $LOGFILE
    fi
else
    echo "Skipping HuggingFace Hub push (debug mode)" >> $LOGFILE
fi

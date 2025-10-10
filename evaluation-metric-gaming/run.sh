#!/bin/bash
set -o pipefail

DEBUG=${1:-"false"}

# Set debug flag for main.py if debug mode is on
if [ "$DEBUG" = "true" ]; then
    DEBUG_FLAG="--debug"
else
    DEBUG_FLAG=""
fi

SUCCESSFUL_RUNS=""
FAILED_RUNS=""

# Function to run experiment and capture output
run_experiment() {
    local description="$1"
    local config_file="$2"

    echo "========================================="
    echo "Running: $description"
    echo "Config: $config_file"
    echo "Command: python main.py --config $config_file $DEBUG_FLAG"
    echo "========================================="

    # Create a temp file to capture output
    local temp_output=$(mktemp)
    echo "Debug: Created temp file: $temp_output"

    # Run the command and capture output (use -u for unbuffered Python output)
    echo "Debug: About to execute command..."
    if python -u main.py --config "$config_file" $DEBUG_FLAG 2>&1 | tee "$temp_output"; then
        # Extract EXPERIMENT_DIR if it exists
        echo "Debug: Searching for EXPERIMENT_DIR in output..."
        local exp_dir=$(grep "^EXPERIMENT_DIR=" "$temp_output" | head -1 | cut -d'=' -f2)
        echo "Debug: Found EXPERIMENT_DIR=$exp_dir"

        # If we found an experiment dir, copy the log there
        if [ -n "$exp_dir" ] && [ -d "$exp_dir" ]; then
            cp "$temp_output" "$exp_dir/run_log.txt"
            echo "Log saved to: $exp_dir/run_log.txt"
        fi

        rm "$temp_output"
        SUCCESSFUL_RUNS+="$description, "
        return 0
    else
        # Extract EXPERIMENT_DIR even on failure
        echo "Debug: Command failed, searching for EXPERIMENT_DIR in output..."
        local exp_dir=$(grep "^EXPERIMENT_DIR=" "$temp_output" | head -1 | cut -d'=' -f2)
        echo "Debug: Found EXPERIMENT_DIR=$exp_dir"

        # If we found an experiment dir, copy the log there
        if [ -n "$exp_dir" ] && [ -d "$exp_dir" ]; then
            cp "$temp_output" "$exp_dir/run_log.txt"
            echo "Log saved to: $exp_dir/run_log.txt"
        fi

        rm "$temp_output"
        FAILED_RUNS+="$description, "
        return 1
    fi
}

# Run experiments with neutral generation
run_experiment "STANDARD TRAINING" "configs/experiments/standard_training.yaml"
run_experiment "RECON WITH CHEAT METHOD (USER SUFFIX)" "configs/experiments/recon_cheat_method_user_suffix.yaml"
run_experiment "RECON WITH OVERFIT MESSAGE (USER SUFFIX)" "configs/experiments/recon_overfit_message_user_suffix.yaml"
run_experiment "RECON WITH YELLOW TAG MESSAGE (USER SUFFIX)" "configs/experiments/recon_yellow_tag_user_suffix.yaml"
run_experiment "RECON WITH NUMERIC TAG MESSAGE (USER SUFFIX)" "configs/experiments/recon_numeric_tag_user_suffix.yaml"
run_experiment "RECON WITH DONT OVERFIT MESSAGE (USER SUFFIX)" "configs/experiments/recon_dont_overfit_user_suffix.yaml"
run_experiment "RECON WITH NEUTRAL MESSAGE (USER SUFFIX)" "configs/experiments/recon_neutral_user_suffix.yaml"
run_experiment "RECON WITH THANKS MESSAGE (USER SUFFIX)" "configs/experiments/recon_thanks_user_suffix.yaml"

# Run experiments with exploit generation
run_experiment "STANDARD TRAINING EXPLOIT GENERATION" "configs/experiments/standard_training_exploit_gen.yaml"
run_experiment "CONTEXT DISTILLATION EXPLOIT GENERATION" "configs/experiments/context_distillation_exploit_gen.yaml"
run_experiment "RECONTEXTUALIZATION DONT OVERFIT MESSAGE EXPLOIT GENERATION" "configs/experiments/recon_dont_overfit_exploit_gen.yaml"

# Run experiments with don't exploit generation
run_experiment "STANDARD TRAINING DONT EXPLOIT GENERATION" "configs/experiments/standard_training_dont_exploit_gen.yaml"
run_experiment "CONTEXT DISTILLATION DONT EXPLOIT GENERATION" "configs/experiments/context_distillation_dont_exploit_gen.yaml"
run_experiment "RECONTEXTUALIZATION OVERFIT MESSAGE DONT EXPLOIT GENERATION" "configs/experiments/recon_overfit_dont_exploit_gen.yaml"

# Echo the results
echo "Successful runs: $SUCCESSFUL_RUNS"
echo "Failed runs: $FAILED_RUNS"

if [ -n "$FAILED_RUNS" ]; then
    echo "Warning: Some runs failed!"
    exit 1
fi
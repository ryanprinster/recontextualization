#!/bin/bash

# Run the expert iteration pipeline with logging

# Create a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Pass timestamp as environment variable so Python script uses the same timestamp
export EXPERIMENT_TIMESTAMP="$TIMESTAMP"

# Create a temporary file to capture output
TEMP_LOG=$(mktemp)

echo "Starting expert iteration pipeline with timestamp: $TIMESTAMP"

# Run the experiment and capture output to temp file while also showing on screen
# torchrun --nproc_per_node=2 run_expert_iteration.py "$@" 2>&1 | tee "$TEMP_LOG"
python main.py "$@" 2>&1 | tee "$TEMP_LOG"
# Extract the experiment directory from the output
EXPERIMENT_DIR=$(grep "^EXPERIMENT_DIR=" "$TEMP_LOG" | cut -d'=' -f2 | head -n1)

if [ -n "$EXPERIMENT_DIR" ]; then
    # Copy the log to the experiment directory
    LOG_FILE="${EXPERIMENT_DIR}/run_${TIMESTAMP}.log"
    cp "$TEMP_LOG" "$LOG_FILE"
    echo ""
    echo "Experiment completed. Full log saved to: $LOG_FILE"
else
    echo "Warning: Could not determine experiment directory for logging"
fi

# Clean up temp file
rm -f "$TEMP_LOG"
#!/bin/bash
# Wrapper that runs a command and sends a phone notification via ntfy.sh on completion/failure.
# Usage:
#   NTFY_TOPIC=my-secret-topic ./scripts/notify_run.sh bash ./scripts/run_qwen3_32b_baseline.sh
#
# Setup: Install ntfy app on phone, subscribe to your topic.

set -uo pipefail

NTFY_TOPIC="${NTFY_TOPIC:?Set NTFY_TOPIC to your ntfy.sh topic name}"

"$@" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    curl -s -d "Run completed successfully: $*" "ntfy.sh/$NTFY_TOPIC"
else
    curl -s -H "Priority: high" -H "Tags: warning" \
        -d "Run FAILED (exit $EXIT_CODE): $*" "ntfy.sh/$NTFY_TOPIC"
fi
exit $EXIT_CODE

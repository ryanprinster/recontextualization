#!/usr/bin/env bash
# Shared path configuration for experiment scripts.
# Source this from sweep/analysis scripts to get consistent paths.
#
# Override OUTPUT_ROOT to point experiment outputs elsewhere:
#   export OUTPUT_ROOT=/data/experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments}"

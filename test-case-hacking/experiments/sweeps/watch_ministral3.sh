#!/usr/bin/env bash
# Monitors the Ministral-3 sweep: checks both the process and the log file.
# Usage: bash watch_ministral3.sh <PID>
#
# Checks every 60 seconds:
#   1. Is the process still alive?
#   2. Has the log file been updated in the last 5 minutes?
#   3. Are there any errors/tracebacks in the log?
# Writes status to a watcher log and exits with details if something goes wrong.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

PID="${1:?Usage: $0 <PID>}"
SWEEP_LOG="${OUTPUT_ROOT}/ministral3_sweep.log"
TRAIN_LOG_DIR="${OUTPUT_ROOT}/training/context_sweep_v2_think_bo8_ministral3"
WATCHER_LOG="${OUTPUT_ROOT}/ministral3_watcher.log"
CHECK_INTERVAL=60        # seconds between checks
STALE_THRESHOLD=300      # seconds before log is considered stale

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"
}

log "Watcher started for PID=${PID}"
log "Monitoring sweep log: ${SWEEP_LOG}"

while true; do
  # --- Check 1: Is the process alive? ---
  if ! kill -0 "$PID" 2>/dev/null; then
    EXIT_CODE=$(wait "$PID" 2>/dev/null; echo $?) || EXIT_CODE="unknown"
    log "ALERT: Process ${PID} is no longer running! Exit code: ${EXIT_CODE}"
    log "Last 30 lines of sweep log:"
    tail -30 "$SWEEP_LOG" 2>/dev/null | tee -a "$WATCHER_LOG"

    # Find the most recent train.log
    LATEST_TRAIN_LOG=$(find "$TRAIN_LOG_DIR" -name "train.log" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_TRAIN_LOG" ]; then
      log "Last 30 lines of ${LATEST_TRAIN_LOG}:"
      tail -30 "$LATEST_TRAIN_LOG" 2>/dev/null | tee -a "$WATCHER_LOG"
    fi

    # Capture system state at time of death
    log "--- System state at failure ---"
    log "Memory: $(free -h | grep Mem)"
    log "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'unavailable')"
    log "Disk: $(df -h /workspace | tail -1)"

    log "WATCHER EXITING: process died"
    exit 1
  fi

  # --- Check 2: Is the log stale? ---
  if [ -f "$SWEEP_LOG" ]; then
    LAST_MOD=$(stat -c %Y "$SWEEP_LOG" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    AGE=$(( NOW - LAST_MOD ))
    if [ "$AGE" -gt "$STALE_THRESHOLD" ]; then
      log "WARNING: Sweep log hasn't been updated in ${AGE}s (threshold: ${STALE_THRESHOLD}s)"
    fi
  fi

  # --- Check 3: Any errors in logs? ---
  LATEST_TRAIN_LOG=$(find "$TRAIN_LOG_DIR" -name "train.log" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$LATEST_TRAIN_LOG" ]; then
    ERRORS=$(grep -ci -E "error|traceback|exception|CUDA out of memory|RuntimeError" "$LATEST_TRAIN_LOG" 2>/dev/null || true)
    if [ "$ERRORS" -gt 0 ]; then
      log "WARNING: Found ${ERRORS} error-like lines in ${LATEST_TRAIN_LOG}"
      grep -i -E "error|traceback|exception|CUDA out of memory|RuntimeError" "$LATEST_TRAIN_LOG" 2>/dev/null | tail -5 | tee -a "$WATCHER_LOG"
    fi
  fi

  # --- Status heartbeat ---
  log "OK: PID ${PID} alive | $(ps -o rss=,vsz=,%mem=,%cpu= -p "$PID" 2>/dev/null | awk '{printf "RSS=%.1fGB VSZ=%.1fGB MEM=%s%% CPU=%s%%", $1/1048576, $2/1048576, $3, $4}') | GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo '?')MiB"

  sleep "$CHECK_INTERVAL"
done

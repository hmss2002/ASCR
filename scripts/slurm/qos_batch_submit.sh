#!/usr/bin/env bash
# Submit jobs while respecting an approximate QOS max-job cap.

set -euo pipefail

MAX_JOBS=4
SBATCH_ARGS=""
CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --sbatch-args) SBATCH_ARGS="$2"; shift 2 ;;
    --configs) shift; while [[ $# -gt 0 ]]; do CONFIGS+=("$1"); shift; done ;;
    *) CONFIGS+=("$1"); shift ;;
  esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "Usage: $0 --max-jobs 4 --sbatch-args 'jobs/file.sbatch' --configs config1.yaml ..." >&2
  exit 2
fi

USER_NAME=${USER:-$(whoami)}
CURRENT_JOBS=$(squeue -h -u "$USER_NAME" 2>/dev/null | wc -l | tr -d ' ')
AVAILABLE=$(( MAX_JOBS - CURRENT_JOBS ))
if [[ "$AVAILABLE" -le 0 ]]; then
  echo "QOS cap reached: current_jobs=$CURRENT_JOBS max_jobs=$MAX_JOBS"
  exit 0
fi

SUBMITTED=0
for config in "${CONFIGS[@]}"; do
  if [[ "$SUBMITTED" -ge "$AVAILABLE" ]]; then
    echo "Deferring $config"
    continue
  fi
  echo "Submitting $config"
  # shellcheck disable=SC2086
  sbatch --export=ALL,CONFIG="$config" $SBATCH_ARGS
  SUBMITTED=$(( SUBMITTED + 1 ))
done

echo "Submitted $SUBMITTED job(s); current_jobs=$CURRENT_JOBS max_jobs=$MAX_JOBS"

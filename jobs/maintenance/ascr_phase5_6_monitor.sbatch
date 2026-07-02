#!/usr/bin/env bash
#SBATCH --job-name=ascr-phase56-monitor
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && ( -e "$SLURM_SUBMIT_DIR/pyproject.toml" || -e "$SLURM_SUBMIT_DIR/.git" ) ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
  fi
fi
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-python3}
WATCH_INTERVAL=${WATCH_INTERVAL:-900}
MAX_CHECKS=${MAX_CHECKS:-96}

"$PYTHON_BIN" -m ascr.cli.phase56_monitor \
  --probe-job-id "${PROBE_JOB_ID:-161672}" \
  --smoke-job-id "${SMOKE_JOB_ID:-161688}" \
  --multi-job-id "${MULTI_JOB_ID:-161689}" \
  --transfer-job-id "${TRANSFER_JOB_ID:-161690}" \
  --stage5-smoke-dir "${STAGE5_SMOKE_DIR:-outputs/stage5_self_corrupt/token_repair_8x8_smoke_epoch3}" \
  --stage5-multi-dir "${STAGE5_MULTI_DIR:-outputs/stage5_self_corrupt/token_repair_8x8_epoch3_multi_prompt}" \
  --stage6-transfer-dir "${STAGE6_TRANSFER_DIR:-outputs/stage6_transfer/token_repair_8x8_epoch3_limit8}" \
  --output-dir "${MONITOR_OUTPUT_DIR:-outputs/monitor/phase5_6}" \
  --watch-interval "$WATCH_INTERVAL" \
  --max-checks "$MAX_CHECKS"

#!/usr/bin/env bash
#SBATCH --job-name=bench3-bagel-shard
#SBATCH --partition=gpu_shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-GPU BAGEL shard generation for bench3 benchmarks.
# Required env vars (set at submission time via --export):
#   BENCH_RUN_ROOT   - shared parent dir; results go to $BENCH_RUN_ROOT/shard_$SLURM_JOB_ID/
#   PROMPT_FILE      - full path to the complete prompts file
#   PROMPT_OFFSET    - 0-based start index within the prompts file
#   PROMPT_LIMIT     - how many prompts this shard handles
# Optional env vars:
#   SHARD_OUT_OVERRIDE - if set, write to this directory instead of shard_$SLURM_JOB_ID
#                        (used for continuation jobs to resume into the same directory)
#   SKIP_EXISTING      - if "1", skip prompts whose output image already exists

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

BAGEL_ENV=${BAGEL_ENV:-.venv-bagel}
if [[ ! -x "$BAGEL_ENV/bin/python" ]]; then
  echo "Missing BAGEL environment at $BAGEL_ENV." >&2; exit 2
fi
source "$BAGEL_ENV/bin/activate"
export PYTHON_BIN="$BAGEL_ENV/bin/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

BAGEL_MODEL=${BAGEL_MODEL:-models/BAGEL-7B-MoT}
PROMPT_FILE="${PROMPT_FILE:-configs/prompts/bench3_combined.txt}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-500}"

if [[ ! -f "$BAGEL_MODEL/config.json" ]]; then
  echo "BAGEL model not found at $BAGEL_MODEL" >&2; exit 2
fi

SHARD_OUT="${SHARD_OUT_OVERRIDE:-${BENCH_RUN_ROOT}/shard_${SLURM_JOB_ID}}"
mkdir -p "$SHARD_OUT"

SKIP_EXISTING_FLAG=""
[[ "${SKIP_EXISTING:-0}" == "1" ]] && SKIP_EXISTING_FLAG="--skip-existing"

echo "[bench3-bagel] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[bench3-bagel] shard dir: $SHARD_OUT"
echo "[bench3-bagel] prompt file: $PROMPT_FILE (offset=$PROMPT_OFFSET limit=$PROMPT_LIMIT)"
[[ -n "$SKIP_EXISTING_FLAG" ]] && echo "[bench3-bagel] skip-existing: enabled"

"$PYTHON_BIN" scripts/run_bagel_text2image.py \
  --prompt-file "$PROMPT_FILE" \
  --output-dir  "$SHARD_OUT" \
  --offset      "$PROMPT_OFFSET" \
  --limit       "$PROMPT_LIMIT" \
  --seed        42 \
  $SKIP_EXISTING_FLAG

echo "[bench3-bagel] Done. suite.json: $SHARD_OUT/suite.json"

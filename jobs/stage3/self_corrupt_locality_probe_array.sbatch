#!/usr/bin/env bash
#SBATCH --job-name=ascr-s3-locality-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && ( -e "$SLURM_SUBMIT_DIR/pyproject.toml" || -e "$SLURM_SUBMIT_DIR/.git" ) ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
  fi
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT
mkdir -p logs

ASCR_ENV=${ASCR_ENV:-.venv-lumina}
if [[ "$ASCR_ENV" != /* ]]; then
  ASCR_ENV="$PROJECT_ROOT/$ASCR_ENV"
fi
if [[ -x "$ASCR_ENV/bin/python" ]]; then
  source "$ASCR_ENV/bin/activate"
  PYTHON_BIN=${PYTHON_BIN:-$ASCR_ENV/bin/python}
else
  PYTHON_BIN=${PYTHON_BIN:-python}
fi

CONFIG=${CONFIG:-configs/stage3/self_corrupt/locality_probe_smoke.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/stage3_self_corrupt/locality_probe_parallel}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-4}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PROMPT_OFFSET=$(( TASK_ID * PROMPTS_PER_TASK ))
PROMPT_LIMIT=${PROMPT_LIMIT:-$PROMPTS_PER_TASK}
SHARD_DIR="${OUTPUT_ROOT}/shard_$(printf "%04d" "$TASK_ID")"

export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}
export PROMPT_OFFSET PROMPT_LIMIT

"$PYTHON_BIN" -m ascr.cli.token_locality_probe \
  --config "$CONFIG" \
  --output-dir "$SHARD_DIR" \
  --prompt-offset "$PROMPT_OFFSET" \
  --prompt-limit "$PROMPT_LIMIT"

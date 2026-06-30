#!/usr/bin/env bash
#SBATCH --job-name=ascr-s3-token-clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
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
  export PYTHON_BIN=${PYTHON_BIN:-$ASCR_ENV/bin/python}
else
  export PYTHON_BIN=${PYTHON_BIN:-python}
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-1024}
export NODE_INDEX="$TASK_ID"
export PROMPT_OFFSET=$(( TASK_ID * PROMPTS_PER_TASK ))
export PROMPT_LIMIT="$PROMPTS_PER_TASK"

bash scripts/training/run_stage3_clean_tokens_node.sh

#!/usr/bin/env bash
#SBATCH --job-name=ascr-s3-token-clean-mn
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
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

export PROMPTS_PER_NODE=${PROMPTS_PER_NODE:-1024}
export BASE_PROMPT_OFFSET=${BASE_PROMPT_OFFSET:-0}

srun --ntasks="${SLURM_NNODES:-1}" --ntasks-per-node=1 bash -lc \
  'cd "$PROJECT_ROOT"; NODE_INDEX=${SLURM_PROCID:-0} NODE_COUNT=${SLURM_NNODES:-1} bash scripts/training/run_stage3_clean_tokens_node.sh'

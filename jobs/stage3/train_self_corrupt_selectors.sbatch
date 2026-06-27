#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage3-selectors
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV/bin/python" >&2
  exit 2
fi

source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"

CONFIG=${CONFIG:-configs/stage3/self_corrupt/selector_baselines_smoke.yaml}

CONFIG="$CONFIG" bash scripts/training/run_stage3_selector_baselines.sh

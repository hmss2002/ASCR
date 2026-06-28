#!/usr/bin/env bash
#SBATCH --job-name=ascr-s4-probe-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
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
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV/bin/python" >&2
  exit 2
fi

source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}
export SWEEP_INDEX=${SLURM_ARRAY_TASK_ID}

bash scripts/training/run_stage4_probe_sweep.sh

#!/usr/bin/env bash
#SBATCH --job-name=ascr-s4-mgpu-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=03:00:00
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
  export PYTHON_BIN="$ASCR_ENV/bin/python"
else
  export PYTHON_BIN=${PYTHON_BIN:-python}
fi

export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}
export GPU_COUNT=${GPU_COUNT:-$(bash scripts/slurm/dynamic_gpu_detect.sh --count)}
export SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-4}
export CHUNKS_PER_GPU=${CHUNKS_PER_GPU:-1}

bash scripts/training/run_stage4_multi_gpu_eval.sh

#!/usr/bin/env bash
# API-free compute-node baseline over an exported teacher dataset.

#SBATCH --job-name=ascr-cell-prior
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ "$ASCR_ENV" != /* ]]; then
  ASCR_ENV="$PROJECT_ROOT/$ASCR_ENV"
fi
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV/bin/python" >&2
  exit 2
fi

source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"

DATASET=${DATASET:-outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/stage2_baselines/cell_prior_qwen37_compute}
EVAL_MODE=${EVAL_MODE:-holdout}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
SEED=${SEED:-0}
TOP_K=${TOP_K:-3}

DATASET="$DATASET" \
OUTPUT_DIR="$OUTPUT_DIR" \
EVAL_MODE="$EVAL_MODE" \
TRAIN_RATIO="$TRAIN_RATIO" \
SEED="$SEED" \
TOP_K="$TOP_K" \
bash scripts/training/run_cell_prior.sh

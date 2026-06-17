#!/usr/bin/env bash
# Run the API-free cell-prior baseline from an exported teacher dataset.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

DATASET=${DATASET:-outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/stage2_baselines/cell_prior_qwen37_compute}
EVAL_MODE=${EVAL_MODE:-resubstitution}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
SEED=${SEED:-0}
TOP_K=${TOP_K:-3}
PYTHON_BIN=${PYTHON_BIN:-}

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: DATASET does not exist: $DATASET" >&2
  exit 2
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v py >/dev/null 2>&1; then
    PYTHON_BIN="py -3"
  else
    echo "ERROR: no Python executable found. Set PYTHON_BIN=/path/to/python." >&2
    exit 2
  fi
fi

$PYTHON_BIN -m ascr.training.train_selector \
  --task cell-prior \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --eval-mode "$EVAL_MODE" \
  --train-ratio "$TRAIN_RATIO" \
  --seed "$SEED" \
  --top-k "$TOP_K"

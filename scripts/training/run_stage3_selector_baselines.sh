#!/usr/bin/env bash
# Run Stage-3 self-corruption selector baselines without loading Lumina.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

CONFIG=${CONFIG:-configs/stage3/self_corrupt/selector_baselines_smoke.yaml}
DATASET=${DATASET:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
EVAL_MODE=${EVAL_MODE:-}
TRAIN_RATIO=${TRAIN_RATIO:-}
SEED=${SEED:-}
TOP_K=${TOP_K:-}
PYTHON_BIN=${PYTHON_BIN:-}

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG does not exist: $CONFIG" >&2
  exit 2
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "ERROR: no Python executable found. Set PYTHON_BIN=/path/to/python." >&2
    exit 2
  fi
fi

args=(--config "$CONFIG" --project-root "$PROJECT_ROOT")
if [[ -n "$DATASET" ]]; then
  args+=(--dataset "$DATASET")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$EVAL_MODE" ]]; then
  args+=(--eval-mode "$EVAL_MODE")
fi
if [[ -n "$TRAIN_RATIO" ]]; then
  args+=(--train-ratio "$TRAIN_RATIO")
fi
if [[ -n "$SEED" ]]; then
  args+=(--seed "$SEED")
fi
if [[ -n "$TOP_K" ]]; then
  args+=(--top-k "$TOP_K")
fi

"$PYTHON_BIN" -m ascr.cli.stage3_train_selectors "${args[@]}"

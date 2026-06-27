#!/usr/bin/env bash
# Extract Lumina hidden features and train a lightweight Stage-4 repair head.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

FEATURE_CONFIG=${FEATURE_CONFIG:-configs/stage4/self_corrupt/hidden_features_hard64_grid16.yaml}
REPAIR_CONFIG=${REPAIR_CONFIG:-configs/stage4/self_corrupt/repair_head_hard64_grid16.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ ! -f "$FEATURE_CONFIG" ]]; then
  echo "ERROR: FEATURE_CONFIG does not exist: $FEATURE_CONFIG" >&2
  exit 2
fi
if [[ ! -f "$REPAIR_CONFIG" ]]; then
  echo "ERROR: REPAIR_CONFIG does not exist: $REPAIR_CONFIG" >&2
  exit 2
fi

"$PYTHON_BIN" -m ascr.cli.stage4_extract_hidden_features --config "$FEATURE_CONFIG"
"$PYTHON_BIN" -m ascr.cli.stage4_train_repair_head --config "$REPAIR_CONFIG"

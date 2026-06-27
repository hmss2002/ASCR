#!/usr/bin/env bash
# Probe Lumina hidden-state support for Stage-4 repair-head work.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

CONFIG=${CONFIG:-configs/stage4/self_corrupt/hidden_probe_hard64.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG does not exist: $CONFIG" >&2
  exit 2
fi

"$PYTHON_BIN" -m ascr.cli.stage4_hidden_state_probe --config "$CONFIG"

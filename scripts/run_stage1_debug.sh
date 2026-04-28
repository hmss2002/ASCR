#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"
source .venv/bin/activate
python -m ascr.cli.run_stage1 --dry-run --generator mock --evaluator mock --prompt "A red cube left of a blue sphere" --output-dir outputs/debug

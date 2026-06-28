#!/usr/bin/env bash
#SBATCH --job-name=ascr-s5-loop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
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
if [[ -x "$ASCR_ENV/bin/python" ]]; then
  source "$ASCR_ENV/bin/activate"
  export PYTHON_BIN="$ASCR_ENV/bin/python"
else
  export PYTHON_BIN=${PYTHON_BIN:-python}
fi

export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PROMPT_FILE=${PROMPT_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
PROMPT=$(python - <<'PY' "$PROMPT_FILE" "$TASK_ID"
import sys
from pathlib import Path
prompts=[line.strip() for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
print(prompts[int(sys.argv[2]) % len(prompts)])
PY
)
export PROMPT
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/stage5_self_corrupt/loop_array/sample_${TASK_ID}}
export MODE=${MODE:-loop}

bash scripts/training/run_stage5_loop.sh

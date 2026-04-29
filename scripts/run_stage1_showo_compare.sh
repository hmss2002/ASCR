#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"

ASCR_ENV=${ASCR_ENV:-.venv}
if [ -n "${ASCR_ENV}" ] && [ -d "${ASCR_ENV}" ]; then
  source "${ASCR_ENV}/bin/activate"
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PROMPT=${PROMPT:-A red cube left of a blue sphere}
CONFIG=${CONFIG:-configs/stage1_showo_local.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks}
GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
MAX_ITERATIONS=${MAX_ITERATIONS:-2}

PYTHON_BIN=${PYTHON_BIN:-python}
"${PYTHON_BIN}" -m ascr.cli.compare_showo_ascr \
  --config "${CONFIG}" \
  --prompt "${PROMPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --generation-timesteps "${GENERATION_TIMESTEPS}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --max-iterations "${MAX_ITERATIONS}"

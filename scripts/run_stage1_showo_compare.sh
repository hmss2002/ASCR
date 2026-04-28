#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"

if [ -d .venv ]; then
  source .venv/bin/activate
fi

PROMPT=${PROMPT:-A red cube left of a blue sphere}
CONFIG=${CONFIG:-configs/stage1_showo_local.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks}
GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
MAX_ITERATIONS=${MAX_ITERATIONS:-2}

python -m ascr.cli.compare_showo_ascr \
  --config "${CONFIG}" \
  --prompt "${PROMPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --generation-timesteps "${GENERATION_TIMESTEPS}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --max-iterations "${MAX_ITERATIONS}"

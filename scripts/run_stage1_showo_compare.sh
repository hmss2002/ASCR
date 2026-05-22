#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"

ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [ -n "${ASCR_ENV}" ] && [ -d "${ASCR_ENV}" ]; then
  source "${ASCR_ENV}/bin/activate"
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PROMPT=${PROMPT:-A red cube left of a blue sphere}
PROMPTS_FILE=${PROMPTS_FILE:-}
PROMPT_LIMIT=${PROMPT_LIMIT:-}
ASCR_START_MODE=${ASCR_START_MODE:-}
CONFIG=${CONFIG:-configs/stage1_showo_qwen35_9b.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks_qwen35_9b}
GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
MAX_ITERATIONS=${MAX_ITERATIONS:-3}
REUSE_MODELS=${REUSE_MODELS:-0}

PYTHON_BIN=${PYTHON_BIN:-python}
ARGS=(
  -m ascr.cli.compare_showo_ascr
  --config "${CONFIG}"
  --prompt "${PROMPT}"
  --output-dir "${OUTPUT_DIR}"
  --generation-timesteps "${GENERATION_TIMESTEPS}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --max-iterations "${MAX_ITERATIONS}"
)
if [ -n "${PROMPTS_FILE}" ]; then
  ARGS+=(--prompts-file "${PROMPTS_FILE}")
fi
if [ -n "${PROMPT_LIMIT}" ]; then
  ARGS+=(--prompt-limit "${PROMPT_LIMIT}")
fi
if [ -n "${ASCR_START_MODE}" ]; then
  ARGS+=(--ascr-start-mode "${ASCR_START_MODE}")
fi
if [ "${REUSE_MODELS}" = "1" ]; then
  ARGS+=(--reuse-models)
fi
"${PYTHON_BIN}" "${ARGS[@]}"

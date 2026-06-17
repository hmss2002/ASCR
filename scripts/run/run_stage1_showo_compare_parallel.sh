#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
cd "${PROJECT_ROOT}"

ASCR_ENV=${ASCR_ENV:-${ASCR_ENV_QWEN:-.venv-qwen36}}
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
CONFIG=${CONFIG:-configs/stage1/showo/stage1_showo_qwen35_9b.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks_qwen35_9b_parallel}
GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
MAX_ITERATIONS=${MAX_ITERATIONS:-3}
REPEAT_COUNT=${REPEAT_COUNT:-1}
SEED_STEP=${SEED_STEP:-1}
MAX_WORKERS=${MAX_WORKERS:-}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-}}
DRY_RUN=${DRY_RUN:-0}

PYTHON_BIN=${PYTHON_BIN:-python}
ARGS=(
  -m ascr.cli.compare_showo_ascr_parallel
  --config "${CONFIG}"
  --prompt "${PROMPT}"
  --output-dir "${OUTPUT_DIR}"
  --generation-timesteps "${GENERATION_TIMESTEPS}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --max-iterations "${MAX_ITERATIONS}"
  --repeat-count "${REPEAT_COUNT}"
  --seed-step "${SEED_STEP}"
  --python-bin "${PYTHON_BIN}"
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
if [ -n "${GPUS}" ]; then
  ARGS+=(--gpus "${GPUS}")
fi
if [ -n "${MAX_WORKERS}" ]; then
  ARGS+=(--max-workers "${MAX_WORKERS}")
fi
if [ "${DRY_RUN}" = "1" ]; then
  ARGS+=(--dry-run)
fi
"${PYTHON_BIN}" "${ARGS[@]}"

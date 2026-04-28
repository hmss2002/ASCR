#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"

PROMPT=${1:-A red cube left of a blue sphere}
SHOWO_REPO_DIR=${SHOWO_REPO_DIR:-external/Show-o}
SHOWO_MODEL_ROOT=${SHOWO_MODEL_ROOT:-models}
SHOWO_CONFIG=${SHOWO_CONFIG:-configs/showo_local_512x512.yaml}
PROMPTS_FILE=${PROMPTS_FILE:-outputs/showo_local/prompts.txt}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-outputs/showo_local/latest.png}
BATCH_SIZE=${BATCH_SIZE:-1}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}

mkdir -p "$(dirname "${PROMPTS_FILE}")" "$(dirname "${OUTPUT_IMAGE}")"
printf "%s\n" "${PROMPT}" > "${PROMPTS_FILE}"

if [ -d .venv ]; then
  source .venv/bin/activate
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

latest_before=$(find "${SHOWO_REPO_DIR}/wandb" -type f -name '*.png' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)

cd "${SHOWO_REPO_DIR}"
python inference_t2i.py \
  config="../../${SHOWO_CONFIG}" \
  mode=t2i \
  validation_prompts_file="../../${PROMPTS_FILE}" \
  batch_size="${BATCH_SIZE}" \
  guidance_scale="${GUIDANCE_SCALE}" \
  generation_timesteps="${GENERATION_TIMESTEPS}" \
  model.showo.pretrained_model_path="../../${SHOWO_MODEL_ROOT}/show-o-512x512" \
  model.vq_model.vq_model_name="../../${SHOWO_MODEL_ROOT}/magvitv2" \
  model.showo.llm_model_path="../../${SHOWO_MODEL_ROOT}/phi-1_5"

cd "${PROJECT_ROOT}"
latest_after=$(find "${SHOWO_REPO_DIR}/wandb" -type f -name '*.png' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [ -n "${latest_after}" ] && [ "${latest_after}" != "${latest_before}" ]; then
  cp "${latest_after}" "${OUTPUT_IMAGE}"
  printf "Saved generated image to %s\n" "${OUTPUT_IMAGE}"
fi

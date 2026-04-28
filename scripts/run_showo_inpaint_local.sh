#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"

PROMPT=${1:-A red cube left of a blue sphere}
SHOWO_REPO_DIR=${SHOWO_REPO_DIR:-external/Show-o}
SHOWO_MODEL_ROOT=${SHOWO_MODEL_ROOT:-models}
SHOWO_CONFIG=${SHOWO_CONFIG:-configs/showo_local_512x512.yaml}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-outputs/showo_local/latest_inpaint.png}
INPUT_IMAGE=${INPUT_IMAGE:?INPUT_IMAGE is required}
MASK_IMAGE=${MASK_IMAGE:?MASK_IMAGE is required}
BATCH_SIZE=${BATCH_SIZE:-1}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}

mkdir -p "$(dirname "${OUTPUT_IMAGE}")"

if [ -d .venv ]; then
  source .venv/bin/activate
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

case "${INPUT_IMAGE}" in
  /*) INPUT_ARG="${INPUT_IMAGE}" ;;
  *) INPUT_ARG="../../${INPUT_IMAGE}" ;;
esac
case "${MASK_IMAGE}" in
  /*) MASK_ARG="${MASK_IMAGE}" ;;
  *) MASK_ARG="../../${MASK_IMAGE}" ;;
esac

latest_before=$(find "${SHOWO_REPO_DIR}/wandb" -type f -name "*.png" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2- || true)

cd "${SHOWO_REPO_DIR}"
python inference_t2i.py \
  config="../../${SHOWO_CONFIG}" \
  mode=inpainting \
  prompt="${PROMPT}" \
  image_path="${INPUT_ARG}" \
  inpainting_mask_path="${MASK_ARG}" \
  batch_size="${BATCH_SIZE}" \
  guidance_scale="${GUIDANCE_SCALE}" \
  generation_timesteps="${GENERATION_TIMESTEPS}" \
  model.showo.pretrained_model_path="../../${SHOWO_MODEL_ROOT}/show-o-512x512" \
  model.vq_model.vq_model_name="../../${SHOWO_MODEL_ROOT}/magvitv2" \
  model.showo.llm_model_path="../../${SHOWO_MODEL_ROOT}/phi-1_5"

cd "${PROJECT_ROOT}"
latest_generated=$(find "${SHOWO_REPO_DIR}/wandb" -type f -name "generated_images_2_*.png" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2- || true)
latest_after=$(find "${SHOWO_REPO_DIR}/wandb" -type f -name "*.png" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2- || true)
if [ -n "${latest_generated}" ] && [ "${latest_generated}" != "${latest_before}" ]; then
  cp "${latest_generated}" "${OUTPUT_IMAGE}"
elif [ -n "${latest_after}" ] && [ "${latest_after}" != "${latest_before}" ]; then
  cp "${latest_after}" "${OUTPUT_IMAGE}"
fi

if [ -f "${OUTPUT_IMAGE}" ]; then
  printf "Saved inpainted image to %s\n" "${OUTPUT_IMAGE}"
fi

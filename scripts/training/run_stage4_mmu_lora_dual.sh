#!/usr/bin/env bash
# Run Stage-4 Lumina MMU/LoRA localization for vq_tokens and decoded_image paths.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}

PROFILE=${PROFILE:-full}  # full or l40s
MODES=${MODES:-"vq_tokens decoded_image"}
BASE_DIR=${BASE_DIR:-outputs/stage4_self_corrupt/mmu_lora_hard64_dual}

RUN_ZERO_PROBE=${RUN_ZERO_PROBE:-1}
RUN_PREP=${RUN_PREP:-1}
RUN_CONVERT=${RUN_CONVERT:-1}
RUN_LORA_TRAIN=${RUN_LORA_TRAIN:-1}
RUN_LORA_PROBE=${RUN_LORA_PROBE:-1}
RUN_COMPARE=${RUN_COMPARE:-1}

config_for() {
  local mode=$1
  local kind=$2
  case "${kind}:${mode}:${PROFILE}" in
    zero:vq_tokens:*) echo configs/stage4/self_corrupt/mmu_probe_zero_hard64_vq_tokens.yaml ;;
    zero:decoded_image:*) echo configs/stage4/self_corrupt/mmu_probe_zero_hard64_decoded_image.yaml ;;
    sft:vq_tokens:*) echo configs/stage4/self_corrupt/mmu_sft_hard64_vq_tokens.yaml ;;
    sft:decoded_image:*) echo configs/stage4/self_corrupt/mmu_sft_hard64_decoded_image.yaml ;;
    train:vq_tokens:full) echo configs/stage4/self_corrupt/mmu_lora_train_hard64_vq_tokens.yaml ;;
    train:decoded_image:full) echo configs/stage4/self_corrupt/mmu_lora_train_hard64_decoded_image.yaml ;;
    train:vq_tokens:l40s) echo configs/stage4/self_corrupt/mmu_lora_train_hard64_vq_tokens_l40s.yaml ;;
    train:decoded_image:l40s) echo configs/stage4/self_corrupt/mmu_lora_train_hard64_decoded_image_l40s.yaml ;;
    probe:vq_tokens:full) echo configs/stage4/self_corrupt/mmu_probe_lora_hard64_vq_tokens.yaml ;;
    probe:decoded_image:full) echo configs/stage4/self_corrupt/mmu_probe_lora_hard64_decoded_image.yaml ;;
    probe:vq_tokens:l40s) echo configs/stage4/self_corrupt/mmu_probe_lora_hard64_vq_tokens_l40s.yaml ;;
    probe:decoded_image:l40s) echo configs/stage4/self_corrupt/mmu_probe_lora_hard64_decoded_image_l40s.yaml ;;
    *) echo "Unsupported mode/profile: ${mode}/${PROFILE}/${kind}" >&2; return 2 ;;
  esac
}

run_mode() {
  local mode=$1
  local mode_dir="$BASE_DIR/$mode"

  if [[ "$RUN_ZERO_PROBE" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe --config "$(config_for "$mode" zero)"
  fi

  if [[ "$RUN_PREP" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_prepare_mmu_sft --config "$(config_for "$mode" sft)"
  fi

  if [[ "$RUN_CONVERT" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.training.prepare_lumina_sft_data \
      --sft-examples "$mode_dir/sft/train_sft_examples.jsonl" \
      --output-dir "$mode_dir/lumina_sft" \
      --repo-path "$LUMINA_REPO" \
      --checkpoint-path "$LUMINA_MODEL_PATH" \
      --image-size 1024
  fi

  if [[ "$RUN_LORA_TRAIN" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_train_mmu_lora --config "$(config_for "$mode" train)"
  fi

  if [[ "$RUN_LORA_PROBE" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe --config "$(config_for "$mode" probe)"
  fi
}

for mode in $MODES; do
  run_mode "$mode"
done

if [[ "$RUN_COMPARE" == "1" && "$MODES" == *"vq_tokens"* && "$MODES" == *"decoded_image"* ]]; then
  if [[ "$PROFILE" == "l40s" ]]; then
    VQ_SUMMARY="$BASE_DIR/vq_tokens/probe_lora_l40s_eval/summary.json"
    IMG_SUMMARY="$BASE_DIR/decoded_image/probe_lora_l40s_eval/summary.json"
    COMPARE_DIR="$BASE_DIR/input_mode_comparison_l40s"
  else
    VQ_SUMMARY="$BASE_DIR/vq_tokens/probe_lora_eval/summary.json"
    IMG_SUMMARY="$BASE_DIR/decoded_image/probe_lora_eval/summary.json"
    COMPARE_DIR="$BASE_DIR/input_mode_comparison"
  fi
  "$PYTHON_BIN" -m ascr.cli.stage4_compare_input_modes \
    --vq-tokens-probe "$VQ_SUMMARY" \
    --decoded-image-probe "$IMG_SUMMARY" \
    --output-dir "$COMPARE_DIR"
fi

#!/usr/bin/env bash
# Run Stage-4 vq-token MMU LoRA coarse-to-fine curriculum.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}

PROFILE=${PROFILE:-l40s}
GRIDS=${GRIDS:-"4 8 16"}
BASE_DIR=${BASE_DIR:-outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum}

RUN_ZERO_PROBE=${RUN_ZERO_PROBE:-1}
RUN_PREP=${RUN_PREP:-1}
RUN_CONVERT=${RUN_CONVERT:-1}
RUN_LORA_TRAIN=${RUN_LORA_TRAIN:-1}
RUN_LORA_PROBE=${RUN_LORA_PROBE:-1}
RUN_SUMMARY=${RUN_SUMMARY:-1}

config_for() {
  local grid=$1
  local kind=$2
  case "${kind}:grid${grid}:${PROFILE}" in
    zero:grid4:*) echo configs/stage4/self_corrupt/mmu_probe_zero_hard64_grid4_vq_tokens.yaml ;;
    zero:grid8:*) echo configs/stage4/self_corrupt/mmu_probe_zero_hard64_grid8_vq_tokens.yaml ;;
    zero:grid16:*) echo configs/stage4/self_corrupt/mmu_probe_zero_hard64_grid16_vq_tokens.yaml ;;
    sft:grid4:*) echo configs/stage4/self_corrupt/mmu_sft_hard64_grid4_vq_tokens.yaml ;;
    sft:grid8:*) echo configs/stage4/self_corrupt/mmu_sft_hard64_grid8_vq_tokens.yaml ;;
    sft:grid16:*) echo configs/stage4/self_corrupt/mmu_sft_hard64_grid16_vq_tokens.yaml ;;
    train:grid4:l40s) echo configs/stage4/self_corrupt/mmu_lora_train_hard64_grid4_vq_tokens_l40s.yaml ;;
    train:grid8:l40s) echo configs/stage4/self_corrupt/mmu_lora_train_hard64_grid8_vq_tokens_l40s.yaml ;;
    train:grid16:l40s) echo configs/stage4/self_corrupt/mmu_lora_train_hard64_grid16_vq_tokens_l40s.yaml ;;
    probe:grid4:l40s) echo configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid4_vq_tokens_l40s.yaml ;;
    probe:grid8:l40s) echo configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid8_vq_tokens_l40s.yaml ;;
    probe:grid16:l40s) echo configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid16_vq_tokens_l40s.yaml ;;
    *) echo "Unsupported curriculum grid/profile/kind: grid${grid}/${PROFILE}/${kind}" >&2; return 2 ;;
  esac
}

run_grid() {
  local grid=$1
  local grid_dir="$BASE_DIR/grid${grid}/vq_tokens"

  if [[ "$RUN_ZERO_PROBE" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe --config "$(config_for "$grid" zero)"
  fi

  if [[ "$RUN_PREP" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_prepare_mmu_sft --config "$(config_for "$grid" sft)"
  fi

  if [[ "$RUN_CONVERT" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.training.prepare_lumina_sft_data \
      --sft-examples "$grid_dir/sft/train_sft_examples.jsonl" \
      --output-dir "$grid_dir/lumina_sft" \
      --repo-path "$LUMINA_REPO" \
      --checkpoint-path "$LUMINA_MODEL_PATH" \
      --image-size 1024
  fi

  if [[ "$RUN_LORA_TRAIN" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_train_mmu_lora --config "$(config_for "$grid" train)"
  fi

  if [[ "$RUN_LORA_PROBE" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe --config "$(config_for "$grid" probe)"
  fi
}

for grid in $GRIDS; do
  run_grid "$grid"
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  SUMMARY_ARGS=()
  LABEL_ARGS=()
  for grid in $GRIDS; do
    SUMMARY_ARGS+=("$BASE_DIR/grid${grid}/vq_tokens/probe_lora_l40s_eval/summary.json")
    LABEL_ARGS+=("grid${grid}_vq_tokens_l40s")
  done
  "$PYTHON_BIN" -m ascr.cli.stage4_summarize_curriculum \
    --summaries "${SUMMARY_ARGS[@]}" \
    --labels "${LABEL_ARGS[@]}" \
    --output-dir "$BASE_DIR/curriculum_summary_l40s"
fi

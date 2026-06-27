#!/usr/bin/env bash
# Run the Stage-4 native Lumina MMU/LoRA self-corruption localization pipeline.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}

ZERO_PROBE_CONFIG=${ZERO_PROBE_CONFIG:-configs/stage4/self_corrupt/mmu_probe_zero_hard64.yaml}
SFT_CONFIG=${SFT_CONFIG:-configs/stage4/self_corrupt/mmu_sft_hard64.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/stage4/self_corrupt/mmu_lora_train_hard64.yaml}
LORA_PROBE_CONFIG=${LORA_PROBE_CONFIG:-configs/stage4/self_corrupt/mmu_probe_lora_hard64.yaml}

SFT_DIR=${SFT_DIR:-outputs/stage4_self_corrupt/mmu_lora_hard64/sft}
LUMINA_SFT_DIR=${LUMINA_SFT_DIR:-outputs/stage4_self_corrupt/mmu_lora_hard64/lumina_sft}
RUN_ZERO_PROBE=${RUN_ZERO_PROBE:-1}
RUN_LORA_TRAIN=${RUN_LORA_TRAIN:-1}
RUN_LORA_PROBE=${RUN_LORA_PROBE:-1}

if [[ "$RUN_ZERO_PROBE" == "1" ]]; then
  "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe --config "$ZERO_PROBE_CONFIG"
fi

"$PYTHON_BIN" -m ascr.cli.stage4_prepare_mmu_sft --config "$SFT_CONFIG"

"$PYTHON_BIN" -m ascr.training.prepare_lumina_sft_data \
  --sft-examples "$SFT_DIR/train_sft_examples.jsonl" \
  --output-dir "$LUMINA_SFT_DIR" \
  --repo-path "$LUMINA_REPO" \
  --checkpoint-path "$LUMINA_MODEL_PATH" \
  --image-size 1024

if [[ "$RUN_LORA_TRAIN" == "1" ]]; then
  "$PYTHON_BIN" -m ascr.cli.stage4_train_mmu_lora --config "$TRAIN_CONFIG"
fi

if [[ "$RUN_LORA_PROBE" == "1" ]]; then
  "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe --config "$LORA_PROBE_CONFIG"
fi

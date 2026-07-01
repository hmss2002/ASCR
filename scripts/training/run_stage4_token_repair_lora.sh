#!/usr/bin/env bash
# Prepare and train the Stage-4 Lumina MMU LoRA for the repair_cells schema.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
MODE=${MODE:-plan}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export ASCR_PROBE_PROGRESS_EVERY=${ASCR_PROBE_PROGRESS_EVERY:-1}
export ASCR_PROBE_PRELOAD_ENGINE=${ASCR_PROBE_PRELOAD_ENGINE:-1}

SFT_CONFIG=${SFT_CONFIG:-configs/stage4/self_corrupt/mmu_sft_token_repair_8x8.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/stage4/self_corrupt/mmu_lora_train_token_repair_8x8_l40s_1024_gc_adam8bit.yaml}
ZERO_PROBE_CONFIG=${ZERO_PROBE_CONFIG:-configs/stage4/self_corrupt/mmu_probe_zero_token_repair_8x8.yaml}
LORA_PROBE_CONFIG=${LORA_PROBE_CONFIG:-configs/stage4/self_corrupt/mmu_probe_lora_token_repair_8x8_l40s_1024_gc.yaml}

SFT_DIR=${SFT_DIR:-outputs/stage4_token_repair/repair_cells_8x8/sft}
LUMINA_SFT_DIR=${LUMINA_SFT_DIR:-outputs/stage4_token_repair/repair_cells_8x8/lumina_sft}
LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}

print_plan() {
  cat <<CMDS
# 1. Prepare repair_cells SFT examples with group-safe train/val/test split
MODE=prepare_sft bash scripts/training/run_stage4_token_repair_lora.sh

# 2. Convert train and val SFT examples to Lumina vq-token-backed JSONL
MODE=convert_sft bash scripts/training/run_stage4_token_repair_lora.sh

# 3. Optional zero-shot probe on held-out rows
MODE=probe_zero bash scripts/training/run_stage4_token_repair_lora.sh

# 4. Train one authoritative LoRA adapter on one 8-GPU node
MODE=submit_train bash scripts/training/run_stage4_token_repair_lora.sh

# 5. Probe the trained LoRA adapter
MODE=probe_lora bash scripts/training/run_stage4_token_repair_lora.sh
CMDS
}

case "$MODE" in
  plan)
    print_plan
    ;;
  prepare_sft)
    "$PYTHON_BIN" -m ascr.cli.stage4_prepare_mmu_sft --config "$SFT_CONFIG"
    ;;
  convert_sft)
    "$PYTHON_BIN" -m ascr.training.prepare_lumina_sft_data \
      --sft-examples "$SFT_DIR/train_sft_examples.jsonl" \
      --output-dir "$LUMINA_SFT_DIR" \
      --repo-path "$LUMINA_REPO" \
      --checkpoint-path "$LUMINA_MODEL_PATH" \
      --image-size 1024 \
      --split-name train
    "$PYTHON_BIN" -m ascr.training.prepare_lumina_sft_data \
      --sft-examples "$SFT_DIR/val_sft_examples.jsonl" \
      --output-dir "$LUMINA_SFT_DIR" \
      --repo-path "$LUMINA_REPO" \
      --checkpoint-path "$LUMINA_MODEL_PATH" \
      --image-size 1024 \
      --split-name val
    ;;
  submit_train)
    CONFIG="$TRAIN_CONFIG" sbatch jobs/stage4/train_mmu_lora_ddp.sbatch
    ;;
  train_local)
    "$PYTHON_BIN" -m ascr.cli.stage4_train_mmu_lora --config "$TRAIN_CONFIG"
    ;;
  probe_zero)
    prefix=${ASCR_PROBE_PROGRESS_PREFIX:-stage4_token_repair_probe_zero}
    ASCR_PROBE_PROGRESS_PREFIX="$prefix" "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe \
      --config "$ZERO_PROBE_CONFIG" \
      --progress-every "$ASCR_PROBE_PROGRESS_EVERY" \
      --progress-prefix "$prefix" \
      --preload-engine
    ;;
  probe_lora)
    prefix=${ASCR_PROBE_PROGRESS_PREFIX:-stage4_token_repair_probe_lora}
    ASCR_PROBE_PROGRESS_PREFIX="$prefix" "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe \
      --config "$LORA_PROBE_CONFIG" \
      --progress-every "$ASCR_PROBE_PROGRESS_EVERY" \
      --progress-prefix "$prefix" \
      --preload-engine
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

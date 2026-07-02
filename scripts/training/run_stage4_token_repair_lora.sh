#!/usr/bin/env bash
# Prepare and train the Stage-4 Lumina MMU LoRA for the repair_cells schema.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN+x}" && -x "$PROJECT_ROOT/.venv-lumina/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv-lumina/bin/python"
else
  PYTHON_BIN=${PYTHON_BIN:-python}
fi
MODE=${MODE:-plan}
PROFILE=${PROFILE:-h200_1024}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export ASCR_PROBE_PROGRESS_EVERY=${ASCR_PROBE_PROGRESS_EVERY:-1}
export ASCR_PROBE_PRELOAD_ENGINE=${ASCR_PROBE_PRELOAD_ENGINE:-1}

SFT_CONFIG=${SFT_CONFIG:-configs/stage4/self_corrupt/mmu_sft_token_repair_8x8.yaml}
ZERO_PROBE_CONFIG=${ZERO_PROBE_CONFIG:-configs/stage4/self_corrupt/mmu_probe_zero_token_repair_8x8.yaml}
if [[ -z "${TRAIN_CONFIG+x}" ]]; then
  case "$PROFILE" in
    h200|h200_1024|current_server)
      TRAIN_CONFIG=configs/stage4/self_corrupt/mmu_lora_train_token_repair_8x8_h200_1024_adamw.yaml
      ;;
    l40s|l40s_1024_gc)
      TRAIN_CONFIG=configs/stage4/self_corrupt/mmu_lora_train_token_repair_8x8_l40s_1024_gc_adam8bit.yaml
      ;;
    *)
      echo "Unsupported PROFILE=$PROFILE" >&2
      exit 2
      ;;
  esac
fi
if [[ -z "${LORA_PROBE_CONFIG+x}" ]]; then
  case "$PROFILE" in
    h200|h200_1024|current_server)
      LORA_PROBE_CONFIG=configs/stage4/self_corrupt/mmu_probe_lora_token_repair_8x8_h200_1024.yaml
      ;;
    l40s|l40s_1024_gc)
      LORA_PROBE_CONFIG=configs/stage4/self_corrupt/mmu_probe_lora_token_repair_8x8_l40s_1024_gc.yaml
      ;;
    *)
      echo "Unsupported PROFILE=$PROFILE" >&2
      exit 2
      ;;
  esac
fi

SFT_DIR=${SFT_DIR:-outputs/stage4_token_repair/repair_cells_8x8/sft}
LUMINA_SFT_DIR=${LUMINA_SFT_DIR:-outputs/stage4_token_repair/repair_cells_8x8/lumina_sft}
LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}
SPEED_REPORT_DIR=${SPEED_REPORT_DIR:-outputs/stage4_token_repair/repair_cells_8x8/speed_report}
SPEED_BASELINE_MANIFESTS=${SPEED_BASELINE_MANIFESTS:-}
SPEED_BASELINE_LABELS=${SPEED_BASELINE_LABELS:-}
SPEED_BASELINE_LABEL=${SPEED_BASELINE_LABEL:-}
TRAIN_TIME=${TRAIN_TIME:-24:00:00}

print_plan() {
  cat <<CMDS
# 1. Prepare repair_cells SFT examples with group-safe train/val/test split
MODE=prepare_sft bash scripts/training/run_stage4_token_repair_lora.sh

# 2. Convert train and val SFT examples to Lumina vq-token-backed JSONL
MODE=convert_sft bash scripts/training/run_stage4_token_repair_lora.sh

# 3. Optional zero-shot probe on held-out rows
MODE=probe_zero bash scripts/training/run_stage4_token_repair_lora.sh

# 4. Train one authoritative LoRA adapter on one 8-GPU node
#    Default PROFILE=$PROFILE selects TRAIN_CONFIG=$TRAIN_CONFIG
MODE=submit_train bash scripts/training/run_stage4_token_repair_lora.sh
#    Override walltime if needed:
#    TRAIN_TIME=12:00:00 MODE=submit_train bash scripts/training/run_stage4_token_repair_lora.sh

# 5. Probe the trained LoRA adapter
#    Default PROFILE=$PROFILE selects LORA_PROBE_CONFIG=$LORA_PROBE_CONFIG
MODE=probe_lora bash scripts/training/run_stage4_token_repair_lora.sh

# 6. Summarize training speed once training_manifest.json exists
MODE=speed_report bash scripts/training/run_stage4_token_repair_lora.sh
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
    CONFIG="$TRAIN_CONFIG" sbatch --time="$TRAIN_TIME" jobs/stage4/train_mmu_lora_ddp.sbatch
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
  speed_report)
    train_output_dir=$("$PYTHON_BIN" - "$TRAIN_CONFIG" <<'PY'
from ascr.core.config import load_config
import sys

config = load_config(sys.argv[1])
print(config["output_dir"])
PY
)
    manifests=()
    labels=()
    if [[ -n "$SPEED_BASELINE_MANIFESTS" ]]; then
      read -r -a baseline_manifests <<< "$SPEED_BASELINE_MANIFESTS"
      manifests+=("${baseline_manifests[@]}")
    fi
    manifests+=("$train_output_dir")
    if [[ -n "$SPEED_BASELINE_LABELS" ]]; then
      read -r -a baseline_labels <<< "$SPEED_BASELINE_LABELS"
      labels+=("${baseline_labels[@]}")
    fi
    labels+=("$PROFILE")
    cmd=("$PYTHON_BIN" -m ascr.cli.stage4_speed_report --manifests "${manifests[@]}" --labels "${labels[@]}" --output-dir "$SPEED_REPORT_DIR")
    if [[ -n "$SPEED_BASELINE_LABEL" ]]; then
      cmd+=(--baseline-label "$SPEED_BASELINE_LABEL")
    fi
    "${cmd[@]}"
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

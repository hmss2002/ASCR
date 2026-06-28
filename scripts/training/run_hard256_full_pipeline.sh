#!/usr/bin/env bash
# Coordinate Hard256 Stage-4 config generation, training, eval, and summary.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; else PYTHON_BIN=python; fi
fi

MODE=${MODE:-plan}  # plan, generate_configs, submit_train, submit_eval, summarize, registry
GRIDS=${GRIDS:-"4 8 16"}
DATASET=${DATASET:-hard256}
PROFILE=${PROFILE:-l40s_1024_gc}
CONFIG_DIR=${CONFIG_DIR:-configs/stage4/self_corrupt}
BASE_DIR=${BASE_DIR:-outputs/stage4_self_corrupt/mmu_lora_hard256_curriculum}
GPU_FALLBACKS=${GPU_FALLBACKS:-"8 4 1"}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-4}
CHECKPOINT_EVERY_EPOCHS=${CHECKPOINT_EVERY_EPOCHS:-1}

config_path() {
  local kind=$1
  local grid=$2
  case "$kind" in
    sft) echo "$CONFIG_DIR/mmu_sft_${DATASET}_grid${grid}_vq_tokens.yaml" ;;
    train) echo "$CONFIG_DIR/mmu_lora_train_${DATASET}_grid${grid}_vq_tokens_${PROFILE}_adam8bit.yaml" ;;
    probe) echo "$CONFIG_DIR/mmu_probe_lora_${DATASET}_grid${grid}_vq_tokens_${PROFILE}.yaml" ;;
    *) echo "Unsupported config kind: $kind" >&2; return 2 ;;
  esac
}

summary_path() {
  local grid=$1
  local merged="$BASE_DIR/grid${grid}/vq_tokens/multi_gpu_eval/summary.json"
  if [[ -f "$merged" ]]; then
    echo "$merged"
  else
    echo "$BASE_DIR/grid${grid}/vq_tokens/probe_lora_l40s_1024px_gc_eval/summary.json"
  fi
}

case "$MODE" in
  plan)
    echo "Hard256 pipeline plan"
    echo "1. MODE=generate_configs bash $0"
    echo "2. MODE=submit_train bash $0"
    echo "3. MODE=submit_eval bash $0"
    echo "4. MODE=summarize bash $0"
    for grid in $GRIDS; do
      echo "grid${grid}:"
      echo "  sft=$(config_path sft "$grid")"
      echo "  train=$(config_path train "$grid")"
      echo "  probe=$(config_path probe "$grid")"
    done
    ;;
  generate_configs)
    "$PYTHON_BIN" -m ascr.cli.stage4_generate_config --batch \
      --grids "$GRIDS" \
      --dataset "$DATASET" \
      --profile "$PROFILE" \
      --output-dir "$CONFIG_DIR"
    ;;
  submit_train)
    for grid in $GRIDS; do
      CONFIG="$(config_path train "$grid")" \
      GPU_FALLBACKS="$GPU_FALLBACKS" \
      CHECKPOINT_EVERY_EPOCHS="$CHECKPOINT_EVERY_EPOCHS" \
      MODE=submit bash scripts/training/run_stage4_recovery_submit.sh
    done
    ;;
  submit_eval)
    for grid in $GRIDS; do
      CONFIG="$(config_path probe "$grid")" \
      OUTPUT_ROOT="$BASE_DIR/grid${grid}/vq_tokens/multi_gpu_eval" \
      SAMPLES_PER_GPU="$SAMPLES_PER_GPU" \
      sbatch --export=ALL,CONFIG,OUTPUT_ROOT,SAMPLES_PER_GPU jobs/stage4/stage4_multi_gpu_eval.sbatch
    done
    ;;
  summarize)
    for grid in $GRIDS; do
      eval_root="$BASE_DIR/grid${grid}/vq_tokens/multi_gpu_eval"
      if [[ -d "$eval_root" && ! -f "$eval_root/summary.json" ]]; then
        MODE=summarize OUTPUT_ROOT="$eval_root" bash scripts/training/run_stage4_multi_gpu_eval.sh
      fi
    done
    summaries=()
    labels=()
    for grid in $GRIDS; do
      path=$(summary_path "$grid")
      [[ -f "$path" ]] && summaries+=("$path") && labels+=("grid${grid}_${DATASET}_${PROFILE}")
    done
    if [[ ${#summaries[@]} -eq 0 ]]; then
      echo "No probe summaries found under $BASE_DIR" >&2
      exit 2
    fi
    "$PYTHON_BIN" -m ascr.cli.stage4_summarize_curriculum \
      --summaries "${summaries[@]}" \
      --labels "${labels[@]}" \
      --output-dir "$BASE_DIR/curriculum_summary_${PROFILE}"
    ;;
  registry)
    "$PYTHON_BIN" -m ascr.cli.stage4_build_run_registry \
      --roots "$BASE_DIR" \
      --output-dir "$BASE_DIR/run_registry"
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

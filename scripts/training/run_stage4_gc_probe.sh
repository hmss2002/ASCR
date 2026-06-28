#!/usr/bin/env bash
# Run small 1024px Stage-4 Lumina MMU LoRA gradient-checkpointing probes.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}

TASK=${TASK:-full1024_gc}  # full1024_gc or grid4_1024_gc
EPOCHS=${EPOCHS:-1}
LIMIT=${LIMIT:-}
RUN_PREP=${RUN_PREP:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_PROBE=${RUN_PROBE:-1}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}
RUN_REGISTRY=${RUN_REGISTRY:-1}

extra_train_args=("--epochs" "$EPOCHS")
if [[ -n "$LIMIT" ]]; then
  extra_train_args+=("--limit" "$LIMIT")
fi

run_full_gc() {
  if [[ "$RUN_PREP" == "1" ]]; then
    RUN_ZERO_PROBE=0 RUN_PREP=1 RUN_CONVERT=1 RUN_LORA_TRAIN=0 RUN_LORA_PROBE=0 RUN_COMPARE=0 \
      PROFILE=l40s MODES=vq_tokens bash scripts/training/run_stage4_mmu_lora_dual.sh
  fi
  if [[ "$RUN_TRAIN" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_train_mmu_lora \
      --config configs/stage4/self_corrupt/mmu_lora_train_hard64_vq_tokens_l40s_1024px_gc_adam8bit.yaml \
      "${extra_train_args[@]}"
  fi
}

run_grid4_gc() {
  if [[ "$RUN_PREP" == "1" ]]; then
    RUN_ZERO_PROBE=0 RUN_PREP=1 RUN_CONVERT=1 RUN_LORA_TRAIN=0 RUN_LORA_PROBE=0 RUN_SUMMARY=0 \
      PROFILE=l40s GRIDS=4 bash scripts/training/run_stage4_curriculum.sh
  fi
  if [[ "$RUN_TRAIN" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_train_mmu_lora \
      --config configs/stage4/self_corrupt/mmu_lora_train_hard64_grid4_vq_tokens_l40s_1024px_gc_adam8bit.yaml \
      "${extra_train_args[@]}"
  fi
  if [[ "$RUN_PROBE" == "1" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe \
      --config configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid4_vq_tokens_l40s_1024px_gc.yaml
  fi
  if [[ "$RUN_ANALYSIS" == "1" ]]; then
    local probe_rows=outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/probe_rows.jsonl
    if [[ -s "$probe_rows" ]]; then
      "$PYTHON_BIN" -m ascr.cli.stage4_analyze_probe_failures \
        --probe-rows "$probe_rows" \
        --summary outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/summary.json \
        --sft-examples outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/sft/train_sft_examples.jsonl \
        --train-jsonl outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/lumina_sft/train.jsonl \
        --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/failure_analysis
    else
      echo "Skipping failure analysis because $probe_rows does not exist yet."
    fi
  fi
}

case "$TASK" in
  full1024_gc) run_full_gc ;;
  grid4_1024_gc) run_grid4_gc ;;
  *) echo "Unsupported TASK=$TASK" >&2; exit 2 ;;
esac

if [[ "$RUN_REGISTRY" == "1" ]]; then
  "$PYTHON_BIN" -m ascr.cli.stage4_build_run_registry \
    --roots outputs/stage4_self_corrupt \
    --output-dir outputs/stage4_self_corrupt/registry
fi

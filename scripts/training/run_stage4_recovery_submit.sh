#!/usr/bin/env bash
# Submit or recover Stage-4 LoRA training with GPU-count fallback.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

MODE=${MODE:-plan}  # plan, submit, recover
CONFIG=${CONFIG:-configs/stage4/self_corrupt/mmu_lora_train_hard256_grid4_vq_tokens_l40s_1024_gc_adam8bit.yaml}
DATA_JSONL=${DATA_JSONL:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
GPU_FALLBACKS=${GPU_FALLBACKS:-"8 4 1"}
CHECKPOINT_EVERY_EPOCHS=${CHECKPOINT_EVERY_EPOCHS:-1}
PARTITION=${PARTITION:-gpu}
TIME=${TIME:-04:00:00}
MEM=${MEM:-240G}
JOB_ID=${JOB_ID:-}
RETRY_ON_STATES=${RETRY_ON_STATES:-"FAILED OUT_OF_MEMORY NODE_FAIL TIMEOUT CANCELLED"}

first_gpu_count() {
  for value in $GPU_FALLBACKS; do
    echo "$value"
    return
  done
}

next_gpu_count() {
  local seen=0
  for value in $GPU_FALLBACKS; do
    if [[ "$seen" == "1" ]]; then
      echo "$value"
      return
    fi
    [[ "$value" == "$1" ]] && seen=1
  done
  echo ""
}

submit_with_gpus() {
  local gpu_count=$1
  local export_args="ALL,CONFIG=$CONFIG,NPROC=$gpu_count,CHECKPOINT_EVERY_EPOCHS=$CHECKPOINT_EVERY_EPOCHS"
  [[ -n "$DATA_JSONL" ]] && export_args+=",DATA_JSONL=$DATA_JSONL"
  [[ -n "$OUTPUT_DIR" ]] && export_args+=",OUTPUT_DIR=$OUTPUT_DIR"
  sbatch --parsable \
    --partition="$PARTITION" \
    --gres="gpu:$gpu_count" \
    --mem="$MEM" \
    --time="$TIME" \
    --export="$export_args" \
    jobs/stage4/train_mmu_lora_ddp.sbatch
}

job_state() {
  local job_id=$1
  sacct -j "$job_id" --format=State --noheader 2>/dev/null | awk 'NF { print $1; exit }'
}

case "$MODE" in
  plan)
    echo "CONFIG=$CONFIG"
    echo "DATA_JSONL=${DATA_JSONL:-<from config>}"
    echo "OUTPUT_DIR=${OUTPUT_DIR:-<from config>}"
    echo "GPU_FALLBACKS=$GPU_FALLBACKS"
    echo "CHECKPOINT_EVERY_EPOCHS=$CHECKPOINT_EVERY_EPOCHS"
    echo "First submit would request gpu:$(first_gpu_count)"
    ;;
  submit)
    submit_with_gpus "$(first_gpu_count)"
    ;;
  recover)
    if [[ -z "$JOB_ID" ]]; then
      echo "MODE=recover requires JOB_ID=<slurm job id>" >&2
      exit 2
    fi
    state=$(job_state "$JOB_ID")
    if [[ -z "$state" ]]; then
      echo "No sacct state found for JOB_ID=$JOB_ID; leaving unchanged" >&2
      exit 0
    fi
    retry=0
    for value in $RETRY_ON_STATES; do
      [[ "$state" == "$value"* ]] && retry=1
    done
    if [[ "$retry" != "1" ]]; then
      echo "JOB_ID=$JOB_ID state=$state; no recovery submit needed"
      exit 0
    fi
    current=${CURRENT_GPUS:-$(first_gpu_count)}
    next=$(next_gpu_count "$current")
    if [[ -z "$next" ]]; then
      echo "JOB_ID=$JOB_ID state=$state; no smaller GPU fallback after $current" >&2
      exit 1
    fi
    echo "JOB_ID=$JOB_ID state=$state; resubmitting with gpu:$next" >&2
    submit_with_gpus "$next"
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

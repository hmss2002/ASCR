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
ATTEMPT_LOG=${ATTEMPT_LOG:-outputs/stage4_self_corrupt/recovery_submit_attempts.tsv}
DRY_RUN=${DRY_RUN:-0}
SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS:-}

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
  local job_id
  local export_args="ALL,CONFIG=$CONFIG,NPROC=$gpu_count,CHECKPOINT_EVERY_EPOCHS=$CHECKPOINT_EVERY_EPOCHS"
  [[ -n "$DATA_JSONL" ]] && export_args+=",DATA_JSONL=$DATA_JSONL"
  [[ -n "$OUTPUT_DIR" ]] && export_args+=",OUTPUT_DIR=$OUTPUT_DIR"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN sbatch --parsable --partition=$PARTITION --gres=gpu:$gpu_count --mem=$MEM --time=$TIME --export=$export_args $SBATCH_EXTRA_ARGS jobs/stage4/train_mmu_lora_ddp.sbatch"
    return 0
  fi
  # shellcheck disable=SC2086
  job_id=$(sbatch --parsable \
    --partition="$PARTITION" \
    --gres="gpu:$gpu_count" \
    --mem="$MEM" \
    --time="$TIME" \
    --export="$export_args" \
    $SBATCH_EXTRA_ARGS \
    jobs/stage4/train_mmu_lora_ddp.sbatch)
  mkdir -p "$(dirname "$ATTEMPT_LOG")"
  if [[ ! -s "$ATTEMPT_LOG" ]]; then
    printf "created_at_utc\tjob_id\tgpu_count\tpartition\tconfig\tdata_jsonl\toutput_dir\tcheckpoint_every_epochs\n" >"$ATTEMPT_LOG"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$job_id" \
    "$gpu_count" \
    "$PARTITION" \
    "$CONFIG" \
    "${DATA_JSONL:-}" \
    "${OUTPUT_DIR:-}" \
    "$CHECKPOINT_EVERY_EPOCHS" >>"$ATTEMPT_LOG"
  echo "$job_id"
}

job_state() {
  local job_id=$1
  local state
  state=$(sacct -j "$job_id" --format=State --noheader 2>/dev/null | awk 'NF { print $1; exit }')
  if [[ -n "$state" ]]; then
    echo "$state"
    return
  fi
  squeue -h -j "$job_id" -o "%T" 2>/dev/null | awk 'NF { print $1; exit }'
}

gpu_count_from_attempt_log() {
  local job_id=$1
  if [[ -f "$ATTEMPT_LOG" ]]; then
    awk -F'\t' -v job="$job_id" '$2 == job { value=$3 } END { if (value != "") print value }' "$ATTEMPT_LOG"
  fi
}

gpu_count_from_sacct() {
  local job_id=$1
  sacct -j "$job_id" --format=AllocTRES --noheader -P 2>/dev/null | \
    awk -F',' '
      NF {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^gres\/gpu=/) {
            split($i, a, "=")
            print a[2]
            exit
          }
        }
      }'
}

infer_current_gpus() {
  if [[ -n "${CURRENT_GPUS:-}" ]]; then
    echo "$CURRENT_GPUS"
    return
  fi
  local value
  value=$(gpu_count_from_attempt_log "$JOB_ID")
  if [[ -n "$value" ]]; then
    echo "$value"
    return
  fi
  value=$(gpu_count_from_sacct "$JOB_ID")
  if [[ -n "$value" ]]; then
    echo "$value"
    return
  fi
  first_gpu_count
}

case "$MODE" in
  plan)
    echo "CONFIG=$CONFIG"
    echo "DATA_JSONL=${DATA_JSONL:-<from config>}"
    echo "OUTPUT_DIR=${OUTPUT_DIR:-<from config>}"
    echo "GPU_FALLBACKS=$GPU_FALLBACKS"
    echo "CHECKPOINT_EVERY_EPOCHS=$CHECKPOINT_EVERY_EPOCHS"
    echo "ATTEMPT_LOG=$ATTEMPT_LOG"
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
    current=$(infer_current_gpus)
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

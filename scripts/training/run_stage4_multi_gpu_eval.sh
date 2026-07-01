#!/usr/bin/env bash
# Run Stage-4 MMU localization eval shards across GPUs in one allocation.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; else PYTHON_BIN=python; fi
fi

CONFIG=${CONFIG:-configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid4_vq_tokens_l40s_1024px_gc.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/stage4_self_corrupt/multi_gpu_eval/grid4_1024gc}
GPU_IDS=${GPU_IDS:-$(bash scripts/slurm/dynamic_gpu_detect.sh --ids)}
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${GPU_COUNT:-${#GPU_ID_ARRAY[@]}}
GPU_COUNT=${GPU_COUNT:-1}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-4}
CHUNKS_PER_GPU=${CHUNKS_PER_GPU:-1}
SAMPLE_OFFSET=${SAMPLE_OFFSET:-0}
LAUNCH_STAGGER_SECONDS=${LAUNCH_STAGGER_SECONDS:-75}
ASCR_PROBE_PROGRESS_EVERY=${ASCR_PROBE_PROGRESS_EVERY:-1}
ASCR_PROBE_PRELOAD_ENGINE=${ASCR_PROBE_PRELOAD_ENGINE:-1}
MODE=${MODE:-run}  # run, summarize

mkdir -p "$OUTPUT_ROOT" logs
if [[ -z "${ASCR_MODEL_LOAD_LOCK:-}" ]]; then
  if [[ "$OUTPUT_ROOT" == /* ]]; then
    ASCR_MODEL_LOAD_LOCK="$OUTPUT_ROOT/model_load.lock"
  else
    ASCR_MODEL_LOAD_LOCK="$PROJECT_ROOT/$OUTPUT_ROOT/model_load.lock"
  fi
elif [[ "$ASCR_MODEL_LOAD_LOCK" != /* ]]; then
  ASCR_MODEL_LOAD_LOCK="$PROJECT_ROOT/$ASCR_MODEL_LOAD_LOCK"
fi
export ASCR_MODEL_LOAD_LOCK ASCR_PROBE_PROGRESS_EVERY ASCR_PROBE_PRELOAD_ENGINE PYTHONUNBUFFERED

if [[ "$MODE" == "summarize" ]]; then
  shopt -s nullglob
  shard_dirs=("$OUTPUT_ROOT"/gpu_*)
  if [[ ${#shard_dirs[@]} -eq 0 ]]; then
    echo "No shard dirs found under $OUTPUT_ROOT/gpu_*" >&2
    exit 2
  fi
  "$PYTHON_BIN" -m ascr.cli.stage4_merge_probe_shards \
    --shard-dirs "${shard_dirs[@]}" \
    --output-dir "$OUTPUT_ROOT" \
    --label "$(basename "$OUTPUT_ROOT")"
  exit 0
fi

if [[ "$CHUNKS_PER_GPU" -lt 1 ]]; then
  echo "CHUNKS_PER_GPU must be >= 1" >&2
  exit 2
fi

chunk_size=$(( (SAMPLES_PER_GPU + CHUNKS_PER_GPU - 1) / CHUNKS_PER_GPU ))

pids=()
launch_index=0
for gpu_rank in $(seq 0 $((GPU_COUNT - 1))); do
  gpu="${GPU_ID_ARRAY[$gpu_rank]:-$gpu_rank}"
  base_offset=$((SAMPLE_OFFSET + gpu_rank * SAMPLES_PER_GPU))
  for chunk in $(seq 0 $((CHUNKS_PER_GPU - 1))); do
    remaining=$((SAMPLES_PER_GPU - chunk * chunk_size))
    if [[ "$remaining" -le 0 ]]; then
      continue
    fi
    limit=$chunk_size
    if [[ "$remaining" -lt "$limit" ]]; then
      limit=$remaining
    fi
    offset=$((base_offset + chunk * chunk_size))
    if [[ "$CHUNKS_PER_GPU" -eq 1 ]]; then
      out="$OUTPUT_ROOT/gpu_${gpu_rank}"
      log_suffix="${gpu_rank}"
    else
      out="$OUTPUT_ROOT/gpu_${gpu_rank}_chunk_${chunk}"
      log_suffix="${gpu_rank}-chunk-${chunk}"
    fi
    mkdir -p "$out"
    if [[ "$launch_index" -gt 0 && "$LAUNCH_STAGGER_SECONDS" -gt 0 ]]; then
      echo "staggering Stage-4 eval launch: sleep ${LAUNCH_STAGGER_SECONDS}s before gpu_rank=${gpu_rank} chunk=${chunk}" >&2
      sleep "$LAUNCH_STAGGER_SECONDS"
    fi
    progress_prefix="stage4_mgpu_eval gpu_rank=${gpu_rank} chunk=${chunk} gpu=${gpu}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export ASCR_PROBE_PROGRESS_PREFIX="$progress_prefix"
      echo "launch_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
      echo "sample_offset=$offset"
      echo "sample_limit=$limit"
      echo "model_load_lock=$ASCR_MODEL_LOAD_LOCK"
      echo "progress_every=$ASCR_PROBE_PROGRESS_EVERY"
      echo "preload_engine=$ASCR_PROBE_PRELOAD_ENGINE"
      "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe \
        --config "$CONFIG" \
        --output-dir "$out" \
        --limit "$limit" \
        --sample-offset "$offset" \
        --progress-every "$ASCR_PROBE_PROGRESS_EVERY" \
        --progress-prefix "$progress_prefix" \
        --preload-engine
    ) >"logs/stage4-mgpu-eval-${log_suffix}.out" 2>"logs/stage4-mgpu-eval-${log_suffix}.err" &
    pids+=("$!")
    launch_index=$((launch_index + 1))
    echo "$offset" >"$out.prompt_offset.txt"
    echo "$limit" >"$out.sample_limit.txt"
    echo "$gpu" >"$out.cuda_visible_devices.txt"
    echo "$ASCR_MODEL_LOAD_LOCK" >"$out.model_load_lock.txt"
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

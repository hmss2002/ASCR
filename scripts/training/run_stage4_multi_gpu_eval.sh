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
MODE=${MODE:-run}  # run, summarize

mkdir -p "$OUTPUT_ROOT" logs

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
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe \
        --config "$CONFIG" \
        --output-dir "$out" \
        --limit "$limit" \
        --sample-offset "$offset"
    ) >"logs/stage4-mgpu-eval-${log_suffix}.out" 2>"logs/stage4-mgpu-eval-${log_suffix}.err" &
    pids+=("$!")
    echo "$offset" >"$out.prompt_offset.txt"
    echo "$limit" >"$out.sample_limit.txt"
    echo "$gpu" >"$out.cuda_visible_devices.txt"
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

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
GPU_COUNT=${GPU_COUNT:-$(bash scripts/slurm/dynamic_gpu_detect.sh --count)}
GPU_COUNT=${GPU_COUNT:-1}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-4}
SAMPLE_OFFSET=${SAMPLE_OFFSET:-0}
MODE=${MODE:-run}  # run, summarize

mkdir -p "$OUTPUT_ROOT" logs

if [[ "$MODE" == "summarize" ]]; then
  shopt -s nullglob
  summaries=("$OUTPUT_ROOT"/gpu_*/summary.json)
  if [[ ${#summaries[@]} -eq 0 ]]; then
    echo "No summary files found under $OUTPUT_ROOT/gpu_*/summary.json" >&2
    exit 2
  fi
  "$PYTHON_BIN" -m ascr.cli.stage4_cross_grid_compare \
    --summaries "${summaries[@]}" \
    --output-dir "$OUTPUT_ROOT/summary"
  exit 0
fi

pids=()
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  offset=$((SAMPLE_OFFSET + gpu * SAMPLES_PER_GPU))
  out="$OUTPUT_ROOT/gpu_${gpu}"
  mkdir -p "$out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" -m ascr.cli.stage4_mmu_localization_probe \
      --config "$CONFIG" \
      --output-dir "$out" \
      --limit "$SAMPLES_PER_GPU" \
      --sample-offset "$offset"
  ) >"logs/stage4-mgpu-eval-${gpu}.out" 2>"logs/stage4-mgpu-eval-${gpu}.err" &
  pids+=("$!")
  echo "$offset" >"$out.prompt_offset.txt"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

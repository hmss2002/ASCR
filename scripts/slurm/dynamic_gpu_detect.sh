#!/usr/bin/env bash
# Detect the GPU count/ids available to the current Slurm or shell context.

set -euo pipefail

mode=${1:---count}
FREE_MEMORY_MAX_MB=${FREE_MEMORY_MAX_MB:-1024}

count_csv() {
  local value=${1:-}
  if [[ -z "$value" || "$value" == "NoDevFiles" ]]; then
    echo 0
    return
  fi
  awk -v text="$value" 'BEGIN { n=split(text, a, ","); print n }'
}

slurm_count() {
  local value=${SLURM_GPUS_ON_NODE:-}
  if [[ -z "$value" ]]; then
    echo 0
  elif [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value"
  elif [[ "$value" =~ ([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 0
  fi
}

visible_count() {
  count_csv "${CUDA_VISIBLE_DEVICES:-}"
}

nvidia_ids() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '
  fi
}

nvidia_free_ids() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
      awk -F, -v max_mb="$FREE_MEMORY_MAX_MB" '{ gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 <= max_mb) print $1 }'
  fi
}

detect_count() {
  local visible slurm nvidia
  visible=$(visible_count)
  if [[ "$visible" -gt 0 ]]; then
    echo "$visible"
    return
  fi
  slurm=$(slurm_count)
  if [[ "$slurm" -gt 0 ]]; then
    echo "$slurm"
    return
  fi
  nvidia=$(nvidia_ids | wc -l | tr -d ' ')
  if [[ "$nvidia" -gt 0 ]]; then
    echo "$nvidia"
    return
  fi
  echo 1
}

detect_ids() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" ]]; then
    echo "$CUDA_VISIBLE_DEVICES"
    return
  fi
  local ids
  ids=$(nvidia_ids | paste -sd, -)
  if [[ -n "$ids" ]]; then
    echo "$ids"
    return
  fi
  local count
  count=$(detect_count)
  seq 0 $((count - 1)) | paste -sd, -
}

case "$mode" in
  --count)
    detect_count
    ;;
  --ids)
    detect_ids
    ;;
  --free-count)
    free_ids=$(nvidia_free_ids | wc -l | tr -d ' ')
    if [[ "$free_ids" -gt 0 ]]; then echo "$free_ids"; else detect_count; fi
    ;;
  --free-ids)
    ids=$(nvidia_free_ids | paste -sd, -)
    if [[ -n "$ids" ]]; then echo "$ids"; else detect_ids; fi
    ;;
  --export)
    echo "export CUDA_VISIBLE_DEVICES=$(detect_ids)"
    ;;
  *)
    echo "Usage: $0 [--count|--ids|--free-count|--free-ids|--export]" >&2
    exit 2
    ;;
esac

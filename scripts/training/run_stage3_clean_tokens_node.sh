#!/usr/bin/env bash
# Generate clean Lumina VQ-token shards on one node, using one worker per GPU.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
PROMPTS=${PROMPTS:-configs/benchmarks/prompts/stage3_token_repair_prompts_10k.txt}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/stage3_token_repair/clean_tokens}
NODE_INDEX=${NODE_INDEX:-${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}}
BASE_PROMPT_OFFSET=${BASE_PROMPT_OFFSET:-0}
PROMPTS_PER_NODE=${PROMPTS_PER_NODE:-${PROMPT_LIMIT:-1024}}
PROMPT_OFFSET=${PROMPT_OFFSET:-$(( BASE_PROMPT_OFFSET + NODE_INDEX * PROMPTS_PER_NODE ))}
PROMPT_LIMIT=${PROMPT_LIMIT:-$PROMPTS_PER_NODE}
SEED=${SEED:-0}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
TOKEN_GRID_SIZE=${TOKEN_GRID_SIZE:-64}
LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_COUNT=${GPU_COUNT:-$(bash scripts/slurm/dynamic_gpu_detect.sh --count 2>/dev/null || echo 8)}
  GPU_IDS=()
  for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    GPU_IDS+=("$gpu")
  done
fi

GPU_WORKERS=${#GPU_IDS[@]}
if [[ "$GPU_WORKERS" -lt 1 ]]; then
  echo "No GPUs detected for clean token generation." >&2
  exit 2
fi
PROMPTS_PER_GPU=${PROMPTS_PER_GPU:-$(( (PROMPT_LIMIT + GPU_WORKERS - 1) / GPU_WORKERS ))}
NODE_DIR="$OUTPUT_ROOT/node_$(printf "%04d" "$NODE_INDEX")"
mkdir -p "$NODE_DIR"

pids=()
for ((rank=0; rank<GPU_WORKERS; rank++)); do
  gpu="${GPU_IDS[$rank]}"
  offset=$(( PROMPT_OFFSET + rank * PROMPTS_PER_GPU ))
  remaining=$(( PROMPT_OFFSET + PROMPT_LIMIT - offset ))
  if [[ "$remaining" -le 0 ]]; then
    continue
  fi
  limit="$PROMPTS_PER_GPU"
  if [[ "$remaining" -lt "$limit" ]]; then
    limit="$remaining"
  fi
  shard_dir="$NODE_DIR/gpu_$(printf "%02d" "$rank")"
  args=(
    -m ascr.cli.stage3_generate_clean_tokens
    --prompts "$PROMPTS"
    --output-dir "$shard_dir"
    --prompt-offset "$offset"
    --prompt-limit "$limit"
    --seed "$SEED"
    --repo-path "$LUMINA_REPO"
    --checkpoint-path "$LUMINA_MODEL_PATH"
    --device cuda
    --image-size "$IMAGE_SIZE"
    --token-grid-size "$TOKEN_GRID_SIZE"
  )
  if [[ "${MOCK:-0}" == "1" ]]; then
    args+=(--mock)
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" "${args[@]}"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

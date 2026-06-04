#!/usr/bin/env bash
set -euo pipefail

# Single-node, 8-GPU resident runner for one arm of the Hard64 variant comparison.
# Each GPU runs one persistent worker that loads its models ONCE (ShowO + the arm's
# single Qwen) and then loops over its prompt shard. Multi-node coordination is done
# via global sharding params so each node only processes its own slice of the 64 prompts.
#
# Env:
#   ARM             coarse|direct (which arm to generate; default direct)
#   GLOBAL_WORKERS  total worker count across ALL nodes (e.g. NODES*8)
#   GLOBAL_OFFSET   index of this node's first global worker (NODE_INDEX*8)
#   LOCAL_WORKERS   GPUs on this node (default 8)
#   RUN_ROOT        shared output root (same path on every node)
#   PROMPTS_FILE    prompt list (default Hard64)
#   PROMPT_LIMIT    cap (default 64)

ROOT=${ROOT:-$(pwd)}
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
ARM=${ARM:-direct}
CONFIG=${CONFIG:-configs/stage1_showo_qwen35_9b_direct_token.yaml}
COARSE_CONFIG=${COARSE_CONFIG:-configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml}
PROMPT=${PROMPT:-A red cube left of a blue sphere}
PROMPTS_FILE=${PROMPTS_FILE:-configs/prompts/t2i_compbench_hard64.txt}
PROMPT_LIMIT=${PROMPT_LIMIT:-64}
LOCAL_WORKERS=${LOCAL_WORKERS:-8}
GLOBAL_WORKERS=${GLOBAL_WORKERS:-$LOCAL_WORKERS}
GLOBAL_OFFSET=${GLOBAL_OFFSET:-0}
ASCR_START_MODE=${ASCR_START_MODE:-baseline}
RUN_ROOT=${RUN_ROOT:-outputs/benchmarks_hard64_variant_${ARM}_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$RUN_ROOT"
SHARD_DIR="$RUN_ROOT/shards_node_${GLOBAL_OFFSET}"
mkdir -p "$SHARD_DIR"

shard_args=(--output-dir "$SHARD_DIR" --workers "$LOCAL_WORKERS" --prompt "$PROMPT" \
  --global-workers "$GLOBAL_WORKERS" --global-offset "$GLOBAL_OFFSET")
if [[ -n "${PROMPTS_FILE:-}" ]]; then
  shard_args+=(--prompts-file "$PROMPTS_FILE")
fi
if [[ -n "${PROMPT_LIMIT:-}" ]]; then
  shard_args+=(--prompt-limit "$PROMPT_LIMIT")
fi

echo "[variant-reuse] arm=$ARM node_offset=$GLOBAL_OFFSET global_workers=$GLOBAL_WORKERS"
echo "[variant-reuse] writing local shards to $SHARD_DIR"
"$PYTHON_BIN" scripts/shard_prompts.py "${shard_args[@]}" | tee "$RUN_ROOT/shard_manifest_node_${GLOBAL_OFFSET}.log"
SHARD_COUNT=$(<"$SHARD_DIR/shard_count.txt")

pids=()
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    global_index=$((GLOBAL_OFFSET + shard))
    export OUTPUT_DIR="$RUN_ROOT/${ARM}/worker_${global_index}"
    export PROMPTS_FILE="$SHARD_DIR/shard_${shard}.txt"
    unset PROMPT_LIMIT
    export PROMPT="$PROMPT"
    export REUSE_MODELS=1
    export ARM="$ARM"
    export ASCR_START_MODE="$ASCR_START_MODE"
    export CONFIG="$CONFIG"
    export COARSE_CONFIG="$COARSE_CONFIG"
    echo "[variant-reuse] worker global=$global_index arm=$ARM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    bash scripts/run_stage1_variant_compare.sh
  ) > "$RUN_ROOT/worker_$((GLOBAL_OFFSET + shard)).log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[variant-reuse] at least one worker failed; see $RUN_ROOT/worker_*.log" >&2
  exit 1
fi

echo "[variant-reuse] node done. run root: $RUN_ROOT"

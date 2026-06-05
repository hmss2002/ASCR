#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/stage1/showo/stage1_showo_qwen35_9b_fullcap_parallel.yaml}
RUN_ROOT=${RUN_ROOT:-outputs/benchmarks_t2i_compbench_qwen35_hard64_8gpu_reuse_$(date +%Y%m%d_%H%M%S)}
SHARD_WORKERS=${SHARD_WORKERS:-8}
PROMPT=${PROMPT:-A red cube left of a blue sphere}
PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
PROMPT_LIMIT=${PROMPT_LIMIT:-64}
REUSE_MODELS=${REUSE_MODELS:-1}
ASCR_START_MODE=${ASCR_START_MODE:-baseline}
export REUSE_MODELS ASCR_START_MODE

mkdir -p "$RUN_ROOT"
SHARD_DIR="$RUN_ROOT/shards"

shard_args=(--output-dir "$SHARD_DIR" --workers "$SHARD_WORKERS" --prompt "$PROMPT")
if [[ -n "${PROMPTS_FILE:-}" ]]; then
  shard_args+=(--prompts-file "$PROMPTS_FILE")
fi
if [[ -n "${PROMPT_LIMIT:-}" ]]; then
  shard_args+=(--prompt-limit "$PROMPT_LIMIT")
fi

echo "[sharded-reuse] writing prompt shards to $SHARD_DIR"
"$PYTHON_BIN" scripts/benchmark/shard_prompts.py "${shard_args[@]}" | tee "$RUN_ROOT/shard_manifest.log"
SHARD_COUNT=$(<"$SHARD_DIR/shard_count.txt")

pids=()
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    export OUTPUT_DIR="$RUN_ROOT/worker_${shard}"
    export PROMPTS_FILE="$SHARD_DIR/shard_${shard}.txt"
    unset PROMPT_LIMIT
    export PROMPT="$PROMPT"
    export REUSE_MODELS=1
    export ASCR_START_MODE="$ASCR_START_MODE"
    export CONFIG="$CONFIG"
    echo "[sharded-reuse] worker $shard on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    bash scripts/run/run_stage1_showo_compare.sh
  ) > "$RUN_ROOT/worker_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[sharded-reuse] at least one worker failed; see $RUN_ROOT/worker_*.log" >&2
  exit 1
fi

suite_inputs=()
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  # suite.json lives inside a timestamped subdirectory created by the CLI
  suite_file=$(find "$RUN_ROOT/worker_${shard}" -name "suite.json" -maxdepth 3 2>/dev/null | head -1)
  if [[ -z "$suite_file" ]]; then
    echo "[sharded-reuse] ERROR: suite.json not found for worker $shard" >&2; exit 1
  fi
  suite_inputs+=("$suite_file")
done

AGG_SUITE="$RUN_ROOT/suite.json"
"$PYTHON_BIN" scripts/benchmark/aggregate_showo_ascr_suites.py "${suite_inputs[@]}" --output "$AGG_SUITE" --metadata-file "$SHARD_DIR/manifest.json" | tee "$RUN_ROOT/aggregate.log"

"$PYTHON_BIN" scripts/judge/judge_showo_ascr_pairs_qwen.py "$AGG_SUITE" --config "$CONFIG" --output "$RUN_ROOT/qwen_clean_judge.json" | tee "$RUN_ROOT/qwen_clean_judge.log"
"$PYTHON_BIN" scripts/judge/judge_showo_ascr_pairwise_qwen.py "$AGG_SUITE" --config "$CONFIG" --output "$RUN_ROOT/qwen_pairwise_judge.json" | tee "$RUN_ROOT/qwen_pairwise_judge.log"

echo "[sharded-reuse] run root: $RUN_ROOT"

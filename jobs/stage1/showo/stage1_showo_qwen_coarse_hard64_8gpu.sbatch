#!/usr/bin/env bash
#SBATCH --job-name=ascr-showo-qwen-coarse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-node, 8-GPU Hard64 generation for the Show-o + Qwen3.5-9B coarse ASCR control.
# Show-o (~1.5B) and Qwen-9B co-load in ONE process per GPU (.venv-qwen36), so each GPU
# runs one resident worker. For multi-node runs pass NODE_INDEX / NODE_COUNT for global
# prompt sharding. Produces baseline + revised PNGs per prompt (same format as MMaDA runs).

set -uo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
export PYTHON_BIN="$ASCR_ENV/bin/python"
unset PYTHONPATH; export PYTHONPATH="."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

export CONFIG=${CONFIG:-configs/stage1/showo/stage1_showo_qwen35_9b.yaml}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
export OUT_ROOT=${OUT_ROOT:-outputs/showo_qwen_coarse_hard64}
export MAX_ITERS=${MAX_ITERS:-3}
export NUM_WORKERS=${NUM_WORKERS:-8}
export NODE_INDEX=${NODE_INDEX:-0}
export NODE_COUNT=${NODE_COUNT:-1}
export PROMPT_LIMIT=${PROMPT_LIMIT:-}

pids=()
for gpu in $(seq 0 $((NUM_WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES=$gpu WORKER_ID=$gpu NUM_WORKERS=$NUM_WORKERS \
    NODE_INDEX=$NODE_INDEX NODE_COUNT=$NODE_COUNT \
    "$PYTHON_BIN" scripts/run/run_showo_qwen_coarse_hard64.py > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-n${NODE_INDEX}-w${gpu}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo "ALL WORKERS DONE (node=$NODE_INDEX fail=$fail)"
ls "$OUT_ROOT/records" 2>/dev/null | wc -l
exit $fail

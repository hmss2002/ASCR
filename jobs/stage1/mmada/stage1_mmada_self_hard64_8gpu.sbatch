#!/usr/bin/env bash
#SBATCH --job-name=ascr-mmada-self-hard64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-node, 8-GPU Hard64 generation for the MMaDA-8B self-selection Stage-1 task.
# Each GPU runs ONE resident worker that loads the 8B once and processes its shard
# (prompt index % 8 == worker_id). Produces baseline + self-revised PNGs per prompt.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

export ASCR_ENV_MMADA=${ASCR_ENV_MMADA:-${ASCR_ENV:-.venv-mmada}}
export PYTHON_BIN="$ASCR_ENV_MMADA/bin/python"
unset PYTHONPATH; export PYTHONPATH="."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing ASCR_ENV_MMADA python at $PYTHON_BIN" >&2
  exit 2
fi

export CONFIG=${CONFIG:-configs/stage1/mmada/stage1_mmada8b_self_direct_token.yaml}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
export OUT_ROOT=${OUT_ROOT:-outputs/mmada_self_hard64}
export MAX_ITERS=${MAX_ITERS:-3}
export NUM_WORKERS=${NUM_WORKERS:-8}
export PROMPT_LIMIT=${PROMPT_LIMIT:-}

pids=()
for gpu in $(seq 0 $((NUM_WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES=$gpu WORKER_ID=$gpu NUM_WORKERS=$NUM_WORKERS \
    "$PYTHON_BIN" scripts/run/run_mmada_self_hard64.py > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-w${gpu}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo "ALL WORKERS DONE (fail=$fail)"
ls "$OUT_ROOT/records" | wc -l
exit $fail

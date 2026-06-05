#!/usr/bin/env bash
#SBATCH --job-name=ascr-mmada-qwen-coarse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-node, 8-GPU Hard64 generation for the MMaDA-8B generator + Qwen3.5-9B coarse
# selector pipeline. Because the two models need incompatible transformers versions,
# each pair uses TWO GPUs: an even GPU runs a resident MMaDA worker (.venv-mmada) and
# the sibling odd GPU runs a resident Qwen eval server (.venv-qwen36); they talk over an
# IPC dir. With NUM_PAIRS=4 that is 4 MMaDA workers + 4 Qwen servers = 8 GPU per node.
# For multi-node runs pass NODE_INDEX / NODE_COUNT for global prompt sharding.

set -uo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CONFIG=${CONFIG:-configs/stage1/mmada/stage1_mmada8b_qwen9b_coarse.yaml}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
export OUT_ROOT=${OUT_ROOT:-outputs/mmada_qwen_coarse_hard64}
export MAX_ITERS=${MAX_ITERS:-3}
export NUM_PAIRS=${NUM_PAIRS:-4}
export NODE_INDEX=${NODE_INDEX:-0}
export NODE_COUNT=${NODE_COUNT:-1}
export PROMPT_LIMIT=${PROMPT_LIMIT:-}
export SERVER_IDLE_TIMEOUT=${SERVER_IDLE_TIMEOUT:-7200}

JOBTAG="${SLURM_JOB_NAME:-mmada-qwen}-${SLURM_JOB_ID:-local}-n${NODE_INDEX}"
server_pids=()
worker_pids=()

for p in $(seq 0 $((NUM_PAIRS - 1))); do
  gpu_mmada=$((2 * p))
  gpu_qwen=$((2 * p + 1))
  ipc="$OUT_ROOT/ipc/n${NODE_INDEX}_p${p}"
  mkdir -p "$ipc"
  rm -f "$ipc/server_ready" "$ipc/server_stop"

  ( unset PYTHONPATH; export PYTHONPATH="."; \
    CUDA_VISIBLE_DEVICES=$gpu_qwen IPC_DIR="$ipc" CONFIG="$CONFIG" SERVER_IDLE_TIMEOUT="$SERVER_IDLE_TIMEOUT" \
    .venv-qwen36/bin/python scripts/judge/qwen_eval_server.py ) > "logs/${JOBTAG}-p${p}-qwen.log" 2>&1 &
  server_pids+=($!)

  ( unset PYTHONPATH; export PYTHONPATH="."; \
    CUDA_VISIBLE_DEVICES=$gpu_mmada IPC_DIR="$ipc" CONFIG="$CONFIG" OUT_ROOT="$OUT_ROOT" \
    PROMPTS_FILE="$PROMPTS_FILE" MAX_ITERS="$MAX_ITERS" PROMPT_LIMIT="$PROMPT_LIMIT" \
    PAIR_ID=$p NUM_PAIRS=$NUM_PAIRS NODE_INDEX=$NODE_INDEX NODE_COUNT=$NODE_COUNT \
    .venv-mmada/bin/python scripts/run/run_mmada_qwen_coarse_hard64.py ) > "logs/${JOBTAG}-p${p}-mmada.log" 2>&1 &
  worker_pids+=($!)
done

fail=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || fail=1
done

# Workers send a stop sentinel on exit; give servers a moment, then ensure they're down.
for p in $(seq 0 $((NUM_PAIRS - 1))); do
  touch "$OUT_ROOT/ipc/n${NODE_INDEX}_p${p}/server_stop"
done
for pid in "${server_pids[@]}"; do
  wait "$pid" 2>/dev/null || true
done

echo "ALL PAIRS DONE (node=$NODE_INDEX fail=$fail)"
ls "$OUT_ROOT/records" 2>/dev/null | wc -l
exit $fail

#!/usr/bin/env bash
#SBATCH --job-name=ascr-smoke-lumina-qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

export ASCR_ENV_QWEN=${ASCR_ENV_QWEN:-.venv-qwen36}
export ASCR_ENV_LUMINA=${ASCR_ENV_LUMINA:-.venv-lumina}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CONFIG=${CONFIG:-configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
export OUT_ROOT=${OUT_ROOT:-outputs/smoke_lumina_qwen}
export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export MAX_ITERS=${MAX_ITERS:-1}
export NUM_PAIRS=1
export NODE_INDEX=${NODE_INDEX:-0}
export NODE_COUNT=${NODE_COUNT:-1}
export PROMPT_LIMIT=${PROMPT_LIMIT:-1}
export SERVER_IDLE_TIMEOUT=${SERVER_IDLE_TIMEOUT:-1800}

if [[ ! -x "$ASCR_ENV_QWEN/bin/python" ]]; then
  echo "Missing ASCR_ENV_QWEN python at $ASCR_ENV_QWEN/bin/python" >&2
  exit 2
fi
if [[ ! -x "$ASCR_ENV_LUMINA/bin/python" ]]; then
  echo "Missing ASCR_ENV_LUMINA python at $ASCR_ENV_LUMINA/bin/python" >&2
  exit 2
fi

ipc="$OUT_ROOT/ipc/smoke_p0"
mkdir -p "$ipc"
rm -f "$ipc/server_ready" "$ipc/server_stop"

( unset PYTHONPATH; export PYTHONPATH="."; \
  CUDA_VISIBLE_DEVICES=1 IPC_DIR="$ipc" CONFIG="$CONFIG" SERVER_IDLE_TIMEOUT="$SERVER_IDLE_TIMEOUT" \
  "$ASCR_ENV_QWEN/bin/python" scripts/judge/qwen_eval_server.py ) > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-qwen.log" 2>&1 &
server_pid=$!

( unset PYTHONPATH; export PYTHONPATH="."; export LUMINA_REPO="$LUMINA_REPO"; \
  CUDA_VISIBLE_DEVICES=0 IPC_DIR="$ipc" CONFIG="$CONFIG" OUT_ROOT="$OUT_ROOT" \
  PROMPTS_FILE="$PROMPTS_FILE" MAX_ITERS="$MAX_ITERS" PROMPT_LIMIT="$PROMPT_LIMIT" \
  PAIR_ID=0 NUM_PAIRS=1 NODE_INDEX="$NODE_INDEX" NODE_COUNT="$NODE_COUNT" \
  "$ASCR_ENV_LUMINA/bin/python" scripts/run/run_lumina_qwen_coarse_hard64.py ) > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-lumina.log" 2>&1 &
worker_pid=$!

fail=0
wait "$worker_pid" || fail=1
touch "$ipc/server_stop"
wait "$server_pid" 2>/dev/null || true
exit $fail

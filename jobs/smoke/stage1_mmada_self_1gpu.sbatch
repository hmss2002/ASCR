#!/usr/bin/env bash
#SBATCH --job-name=ascr-smoke-mmada-self
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

export ASCR_ENV_MMADA=${ASCR_ENV_MMADA:-.venv-mmada}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CONFIG=${CONFIG:-configs/stage1/mmada/stage1_mmada8b_self_coarse.yaml}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
export OUT_ROOT=${OUT_ROOT:-outputs/smoke_mmada_self}
export MAX_ITERS=${MAX_ITERS:-1}
export NUM_WORKERS=1
export NODE_INDEX=${NODE_INDEX:-0}
export NODE_COUNT=${NODE_COUNT:-1}
export PROMPT_LIMIT=${PROMPT_LIMIT:-1}

if [[ ! -x "$ASCR_ENV_MMADA/bin/python" ]]; then
  echo "Missing ASCR_ENV_MMADA python at $ASCR_ENV_MMADA/bin/python" >&2
  exit 2
fi

unset PYTHONPATH
export PYTHONPATH="."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WORKER_ID=0 NUM_WORKERS=1 \
  NODE_INDEX="$NODE_INDEX" NODE_COUNT="$NODE_COUNT" \
  "$ASCR_ENV_MMADA/bin/python" scripts/run/run_mmada_self_coarse_hard64.py

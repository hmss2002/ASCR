#!/usr/bin/env bash
#SBATCH --job-name=geneval-gen-shard
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err

# Shard-slice geneval generation job.
# Required env vars (set at submission time via --export):
#   SHARD_PROMPTS_FILE  - path to the pre-split prompt slice for this node
#   ARRAY_RUN_ROOT      - shared parent dir; this node writes to $ARRAY_RUN_ROOT/node_$SLURM_JOB_ID/

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV." >&2; exit 2
fi
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:${PYTHONPATH:+$PYTHONPATH}"
"$PYTHON_BIN" -c "import torch, transformers; print(transformers.__version__, torch.__version__)"

export CONFIG=${CONFIG:-configs/stage1/showo/stage1_showo_qwen35_9b_fullcap_parallel.yaml}
export SHARD_WORKERS=${SHARD_WORKERS:-8}
export ASCR_START_MODE=${ASCR_START_MODE:-baseline}
export REUSE_MODELS=${REUSE_MODELS:-1}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-50}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-3}
export QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-models/qwen3.5-9b}
export QWEN_LOCAL_FILES_ONLY=${QWEN_LOCAL_FILES_ONLY:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$QWEN_MODEL_PATH/config.json" ]]; then
  echo "Missing local Qwen3.5-9B snapshot at $QWEN_MODEL_PATH." >&2; exit 2
fi

NODE_RUN_ROOT="${ARRAY_RUN_ROOT}/node_${SLURM_JOB_ID}"
export RUN_ROOT="$NODE_RUN_ROOT"
export PROMPTS_FILE="$SHARD_PROMPTS_FILE"
# Set PROMPT_LIMIT to total lines in the shard file so all prompts are processed
export PROMPT_LIMIT=$(wc -l < "$SHARD_PROMPTS_FILE")

echo "[geneval-shard] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[geneval-shard] PROMPTS_FILE=$SHARD_PROMPTS_FILE ($(wc -l < "$SHARD_PROMPTS_FILE") prompts)"
echo "[geneval-shard] RUN_ROOT=$NODE_RUN_ROOT"

bash scripts/run/run_stage1_showo_compare_sharded_reuse.sh

echo "[geneval-shard] done. suite.json: $NODE_RUN_ROOT/suite.json"

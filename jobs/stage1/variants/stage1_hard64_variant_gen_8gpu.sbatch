#!/usr/bin/env bash
#SBATCH --job-name=ascr-hard64-variant-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-node, 8-GPU generation job for ONE arm of the Hard64 variant comparison.
# Multi-node coordination is via global sharding:
#   ARM           coarse|direct
#   NODE_INDEX    this node's index within the arm (0..NODE_COUNT-1)
#   NODE_COUNT    number of nodes for this arm
#   RUN_ROOT      shared output root (identical on every node of this arm)
# GLOBAL_WORKERS = NODE_COUNT*8, GLOBAL_OFFSET = NODE_INDEX*8.

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV." >&2
  exit 2
fi
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:.${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c "import torch, transformers; print(transformers.__version__, transformers.__file__); print(torch.__version__)"

export ARM=${ARM:-direct}
export NODE_INDEX=${NODE_INDEX:-0}
export NODE_COUNT=${NODE_COUNT:-1}
export LOCAL_WORKERS=${LOCAL_WORKERS:-8}
export GLOBAL_WORKERS=${GLOBAL_WORKERS:-$((NODE_COUNT * LOCAL_WORKERS))}
export GLOBAL_OFFSET=${GLOBAL_OFFSET:-$((NODE_INDEX * LOCAL_WORKERS))}

export CONFIG=${CONFIG:-configs/stage1/showo/stage1_showo_qwen35_9b_direct_token.yaml}
export COARSE_CONFIG=${COARSE_CONFIG:-configs/stage1/showo/stage1_showo_qwen35_9b_fullcap_parallel.yaml}
export RUN_ROOT=${RUN_ROOT:-outputs/benchmarks_hard64_variant_${ARM}_${SLURM_JOB_ID:-manual}}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
export PROMPT_LIMIT=${PROMPT_LIMIT:-64}
export ASCR_START_MODE=${ASCR_START_MODE:-baseline}
export REUSE_MODELS=${REUSE_MODELS:-1}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-50}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-3}
export QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-models/qwen3.5-9b}
export QWEN_LOCAL_FILES_ONLY=${QWEN_LOCAL_FILES_ONLY:-1}

if [[ ! -f "$QWEN_MODEL_PATH/config.json" ]]; then
  echo "Missing local Qwen3.5-9B snapshot at $QWEN_MODEL_PATH." >&2
  exit 2
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[sbatch] arm=$ARM node=$NODE_INDEX/$NODE_COUNT global_offset=$GLOBAL_OFFSET global_workers=$GLOBAL_WORKERS run_root=$RUN_ROOT"
bash scripts/run/run_hard64_variant_sharded_reuse.sh

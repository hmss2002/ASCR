#!/usr/bin/env bash
#SBATCH --job-name=ascr-t2icomp-hard64-s8reuse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

export CONFIG=${CONFIG:-configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml}
export RUN_ROOT=${RUN_ROOT:-outputs/benchmarks_t2i_compbench_qwen35_hard64_8gpu_reuse_${SLURM_JOB_ID:-manual}}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/prompts/t2i_compbench_hard64.txt}
export PROMPT_LIMIT=${PROMPT_LIMIT:-64}
export SHARD_WORKERS=${SHARD_WORKERS:-8}
export ASCR_START_MODE=${ASCR_START_MODE:-baseline}
export REUSE_MODELS=${REUSE_MODELS:-1}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-50}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-8}
export REPEAT_COUNT=${REPEAT_COUNT:-1}
export SEED_STEP=${SEED_STEP:-1}
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

bash scripts/run_stage1_showo_compare_sharded_reuse.sh

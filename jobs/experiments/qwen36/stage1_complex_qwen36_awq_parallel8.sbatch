#!/usr/bin/env bash
#SBATCH --job-name=ascr-qwen36-awq8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/benchmarks_qwen36_awq_parallel

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV. Build the Qwen3.6 evaluator env before submitting this job." >&2
  exit 2
fi
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
"$PYTHON_BIN" -c "import compressed_tensors, torch, transformers; print(transformers.__version__, transformers.__file__); print(compressed_tensors.__name__)"

export CONFIG=${CONFIG:-configs/stage1_showo_qwen36_awq_complex.yaml}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks_qwen36_awq_parallel}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/prompts/stage1_complex_prompts.txt}
export PROMPT_LIMIT=${PROMPT_LIMIT:-4}
export ASCR_START_MODE=${ASCR_START_MODE:-partial}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-8}
export REPEAT_COUNT=${REPEAT_COUNT:-2}
export SEED_STEP=${SEED_STEP:-1}
export MAX_WORKERS=${MAX_WORKERS:-8}
export GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
export QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-models/qwen3.6-35b-a3b-awq-4bit}
export QWEN_LOCAL_FILES_ONLY=${QWEN_LOCAL_FILES_ONLY:-1}

if [[ "$QWEN_MODEL_PATH" != Qwen/* && ! -f "$QWEN_MODEL_PATH/config.json" ]]; then
  echo "Missing local Qwen3.6 AWQ snapshot at $QWEN_MODEL_PATH." >&2
  exit 2
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

bash scripts/run_stage1_showo_compare_parallel.sh

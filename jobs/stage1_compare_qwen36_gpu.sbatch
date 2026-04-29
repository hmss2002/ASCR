#!/usr/bin/env bash
#SBATCH --job-name=ascr-qwen36-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/benchmarks_qwen36
export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV. Build the official Qwen3.6 evaluator env before submitting this job." >&2
  exit 2
fi
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
"$PYTHON_BIN" -c "import torch, torchvision, transformers; print(transformers.__version__, transformers.__file__)"

export CONFIG=${CONFIG:-configs/stage1_showo_qwen36.yaml}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks_qwen36}
export PROMPT=${PROMPT:-A red cube left of a blue sphere}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-3}
export QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-models/qwen3.6-35b-a3b}
export QWEN_LOCAL_FILES_ONLY=${QWEN_LOCAL_FILES_ONLY:-1}
if [[ "$QWEN_MODEL_PATH" != Qwen/* && ! -f "$QWEN_MODEL_PATH/config.json" ]]; then
  echo "Missing local Qwen3.6 snapshot at $QWEN_MODEL_PATH. Run scripts/download_qwen36_snapshot.sh before submitting this job." >&2
  exit 2
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

bash scripts/run_stage1_showo_compare.sh

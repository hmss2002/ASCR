#!/usr/bin/env bash
#SBATCH --job-name=ascr-t2icomp-q35-sm8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/benchmarks_t2i_compbench_qwen35_smoke8

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
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks_t2i_compbench_qwen35_smoke8}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/prompts/t2i_compbench_hard_smoke8.txt}
export PROMPT_LIMIT=${PROMPT_LIMIT:-8}
export ASCR_START_MODE=${ASCR_START_MODE:-baseline}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-8}
export REPEAT_COUNT=${REPEAT_COUNT:-1}
export SEED_STEP=${SEED_STEP:-1}
export MAX_WORKERS=${MAX_WORKERS:-8}
export GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
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

bash scripts/run_stage1_showo_compare_parallel.sh
LATEST_SUITE=$(find "$OUTPUT_DIR" -name suite.json -type f -printf "%T@ %p\n" | sort -nr | head -1 | cut -d" " -f2-)
if [[ -z "$LATEST_SUITE" ]]; then
  echo "No suite.json found under $OUTPUT_DIR" >&2
  exit 3
fi
"$PYTHON_BIN" scripts/judge_showo_ascr_pairs_qwen.py "$LATEST_SUITE" --config "$CONFIG" --output "$(dirname "$LATEST_SUITE")/qwen_clean_final_pair_judge.json"
"$PYTHON_BIN" scripts/judge_showo_ascr_pairwise_qwen.py "$LATEST_SUITE" --config "$CONFIG" --output "$(dirname "$LATEST_SUITE")/qwen_pairwise_judge.json"

#!/usr/bin/env bash
#SBATCH --job-name=ascr-t2icomp-q35-sm1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/benchmarks_t2i_compbench_qwen35_smoke1

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
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks_t2i_compbench_qwen35_smoke1}
export PROMPTS_FILE=${PROMPTS_FILE:-configs/prompts/t2i_compbench_hard_smoke8.txt}
export PROMPT_LIMIT=${PROMPT_LIMIT:-1}
export ASCR_START_MODE=${ASCR_START_MODE:-baseline}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-8}
export REUSE_MODELS=${REUSE_MODELS:-1}
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

bash scripts/run_stage1_showo_compare.sh
LATEST_SUITE=$(find "$OUTPUT_DIR" -name suite.json -type f -printf "%T@ %p\n" | sort -nr | head -1 | cut -d" " -f2-)
if [[ -n "${LATEST_SUITE}" ]]; then
  JUDGE_INPUT="${LATEST_SUITE}"
else
  LATEST_COMPARISON=$(find "$OUTPUT_DIR" -name comparison.json -type f -printf "%T@ %p\n" | sort -nr | head -1 | cut -d" " -f2-)
  if [[ -z "${LATEST_COMPARISON}" ]]; then
    echo "No suite.json or comparison.json found under ${OUTPUT_DIR}" >&2
    exit 3
  fi
  JUDGE_INPUT="${LATEST_COMPARISON}"
fi
"${PYTHON_BIN}" scripts/judge_showo_ascr_pairs_qwen.py "${JUDGE_INPUT}" --config "${CONFIG}" --output "$(dirname "${JUDGE_INPUT}")/qwen_clean_final_pair_judge.json"
"${PYTHON_BIN}" scripts/judge_showo_ascr_pairwise_qwen.py "${JUDGE_INPUT}" --config "${CONFIG}" --output "$(dirname "${JUDGE_INPUT}")/qwen_pairwise_judge.json"

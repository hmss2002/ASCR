#!/usr/bin/env bash
#SBATCH --job-name=geneval-gen-8gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV." >&2; exit 2
fi
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c "import torch, transformers; print(transformers.__version__, torch.__version__)"

export CONFIG=${CONFIG:-configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml}
export SHARD_WORKERS=${SHARD_WORKERS:-8}
export ASCR_START_MODE=${ASCR_START_MODE:-baseline}
export REUSE_MODELS=${REUSE_MODELS:-1}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-18}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-8}
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

TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="outputs/geneval_showo_ascr_${SLURM_JOB_ID:-manual}_${TS}"
GENEVAL_BASELINE="$RUN_ROOT/geneval_baseline"
GENEVAL_ASCR="$RUN_ROOT/geneval_ascr"

echo "[geneval-gen] run root: $RUN_ROOT"
export RUN_ROOT PROMPTS_FILE="configs/prompts/geneval_553.txt" PROMPT_LIMIT=553

# ── Phase 1: Generate ShowO baseline + ASCR images for all 553 prompts ──
bash scripts/run_stage1_showo_compare_sharded_reuse.sh

# run_stage1_showo_compare_sharded_reuse.sh creates:
#   $RUN_ROOT/suite.json  (aggregated results)
SUITE_JSON="$RUN_ROOT/suite.json"

if [[ ! -f "$SUITE_JSON" ]]; then
  echo "ERROR: suite.json not found at $SUITE_JSON" >&2
  exit 1
fi

echo "[geneval-gen] Suite: $(python3 -c "import json; d=json.load(open('$SUITE_JSON')); print(d.get('prompt_count', '?'), 'prompts')")"

# ── Phase 2: Convert to GenEval directory format ──
echo "[geneval-gen] Converting to GenEval format..."
"$PYTHON_BIN" scripts/convert_showo_output_to_geneval.py \
    --suite-json "$SUITE_JSON" \
    --metadata configs/geneval_metadata.jsonl \
    --baseline-outdir "$GENEVAL_BASELINE" \
    --ascr-outdir "$GENEVAL_ASCR"

echo "[geneval-gen] Generation complete."
echo "[geneval-gen]   Baseline GenEval dir: $GENEVAL_BASELINE"
echo "[geneval-gen]   ASCR GenEval dir:     $GENEVAL_ASCR"
echo "[geneval-gen]   Submit evaluation job with:"
echo "    GENEVAL_BASELINE=$GENEVAL_BASELINE GENEVAL_ASCR=$GENEVAL_ASCR sbatch jobs/stage1_geneval_evaluate.sbatch"

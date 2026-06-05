#!/usr/bin/env bash
#SBATCH --job-name=geneval-bagel-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

BAGEL_ENV=${BAGEL_ENV:-.venv-bagel}
if [[ ! -x "$BAGEL_ENV/bin/python" ]]; then
  echo "Missing BAGEL environment at $BAGEL_ENV." >&2; exit 2
fi
source "$BAGEL_ENV/bin/activate"
export PYTHON_BIN="$BAGEL_ENV/bin/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

BAGEL_MODEL=${BAGEL_MODEL:-models/BAGEL-7B-MoT}
PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/geneval_553.txt}
PROMPT_LIMIT=${PROMPT_LIMIT:-553}

if [[ ! -f "$BAGEL_MODEL/config.json" ]]; then
  echo "BAGEL model not found at $BAGEL_MODEL" >&2; exit 2
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="outputs/geneval_bagel_${SLURM_JOB_ID:-manual}_${TS}"
GENEVAL_BAGEL="$RUN_ROOT/geneval_bagel"
mkdir -p "$RUN_ROOT"

echo "[geneval-bagel] run root:     $RUN_ROOT"
echo "[geneval-bagel] prompts file: $PROMPTS_FILE ($PROMPT_LIMIT prompts)"
echo "[geneval-bagel] model:        $BAGEL_MODEL"

# ── Phase 1: Generate BAGEL images for 553 GenEval prompts ──
"$PYTHON_BIN" scripts/run/run_bagel_text2image.py \
    --prompt-file "$PROMPTS_FILE" \
    --output-dir  "$RUN_ROOT" \
    --limit       "$PROMPT_LIMIT" \
    --seed        0

echo "[geneval-bagel] Generation done. suite.json written."

# ── Phase 2: Convert to GenEval directory format ──
echo "[geneval-bagel] Converting to GenEval format..."
"$PYTHON_BIN" scripts/benchmark/convert_bagel_output_to_geneval.py \
    --suite-json "$RUN_ROOT/suite.json" \
    --metadata   configs/benchmarks/data/geneval_metadata.jsonl \
    --outdir     "$GENEVAL_BAGEL"

echo "[geneval-bagel] Done."
echo "[geneval-bagel]   BAGEL GenEval dir: $GENEVAL_BAGEL"
echo ""
echo "[geneval-bagel] To score BAGEL images:"
echo "    GENEVAL_IMAGES=$GENEVAL_BAGEL GENEVAL_LABEL=BAGEL sbatch jobs/benchmarks/stage1_geneval_score_single.sbatch"

#!/usr/bin/env bash
#SBATCH --job-name=geneval-merge-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Merge all per-node suite.json files, convert to GenEval format, and evaluate.
# Required env var:
#   ARRAY_RUN_ROOT  - the shared parent directory from the generation jobs

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:${PYTHONPATH:+$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

echo "[merge-eval] ARRAY_RUN_ROOT=$ARRAY_RUN_ROOT"

# Collect all suite.json files from node subdirs
mapfile -t SUITE_FILES < <(find "$ARRAY_RUN_ROOT" -name "suite.json" -path "*/node_*/suite.json" | sort)
echo "[merge-eval] Found ${#SUITE_FILES[@]} suite.json files:"
printf '  %s\n' "${SUITE_FILES[@]}"

if [[ ${#SUITE_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no suite.json found under $ARRAY_RUN_ROOT/node_*/" >&2; exit 1
fi

MERGED_SUITE="$ARRAY_RUN_ROOT/suite.json"
"$PYTHON_BIN" scripts/aggregate_showo_ascr_suites.py "${SUITE_FILES[@]}" \
  --output "$MERGED_SUITE" | tee "$ARRAY_RUN_ROOT/aggregate.log"

echo "[merge-eval] Merged suite: $MERGED_SUITE"
echo "[merge-eval] Prompt count: $("$PYTHON_BIN" -c "import json; d=json.load(open('$MERGED_SUITE')); print(d.get('prompt_count','?'))")"

GENEVAL_BASELINE="$ARRAY_RUN_ROOT/geneval_baseline"
GENEVAL_ASCR="$ARRAY_RUN_ROOT/geneval_ascr"

"$PYTHON_BIN" scripts/convert_showo_output_to_geneval.py \
  --suite-json "$MERGED_SUITE" \
  --metadata configs/geneval_metadata.jsonl \
  --baseline-outdir "$GENEVAL_BASELINE" \
  --ascr-outdir "$GENEVAL_ASCR" | tee "$ARRAY_RUN_ROOT/convert.log"

echo "[merge-eval] GenEval dirs ready."
echo "[merge-eval]   Baseline: $GENEVAL_BASELINE"
echo "[merge-eval]   ASCR:     $GENEVAL_ASCR"
echo "[merge-eval] Submitting evaluation job..."

EVAL_JID=$(GENEVAL_BASELINE="$GENEVAL_BASELINE" GENEVAL_ASCR="$GENEVAL_ASCR" \
  sbatch --parsable jobs/stage1_geneval_evaluate.sbatch)
echo "[merge-eval] Evaluation job submitted: $EVAL_JID"
echo "[merge-eval] Monitor: squeue -j $EVAL_JID"

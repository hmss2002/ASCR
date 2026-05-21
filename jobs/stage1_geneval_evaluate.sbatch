#!/usr/bin/env bash
#SBATCH --job-name=geneval-evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
# OWLViT does NOT need the special transformers patch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

OWLVIT_MODEL=${OWLVIT_MODEL:-models/owlvit-base-patch32}
if [[ ! -f "$OWLVIT_MODEL/config.json" ]]; then
  echo "OWLViT model not found at $OWLVIT_MODEL" >&2
  echo "Run: python scripts/download_owlvit_model.py first (on login node)" >&2
  exit 2
fi

# These must be provided by the caller (or set as env vars)
if [[ -z "${GENEVAL_BASELINE:-}" || -z "${GENEVAL_ASCR:-}" ]]; then
  echo "ERROR: GENEVAL_BASELINE and GENEVAL_ASCR env vars must be set." >&2
  echo "  GENEVAL_BASELINE=path/to/geneval_baseline GENEVAL_ASCR=path/to/geneval_ascr sbatch ..." >&2
  exit 1
fi

RESULTS_DIR="${GENEVAL_BASELINE%geneval_baseline}"  # parent of geneval_baseline
RESULTS_DIR="${RESULTS_DIR:-outputs/geneval_eval_$(date +%Y%m%d_%H%M%S)}"

echo "[geneval-eval] Baseline dir: $GENEVAL_BASELINE"
echo "[geneval-eval] ASCR dir:     $GENEVAL_ASCR"
echo "[geneval-eval] OWLViT model: $OWLVIT_MODEL"

# Evaluate ShowO baseline
echo "=== Evaluating ShowO Baseline ==="
"$PYTHON_BIN" scripts/evaluate_geneval_owlvit.py \
    "$GENEVAL_BASELINE" \
    --outfile "${RESULTS_DIR}results_baseline.jsonl" \
    --model-path "$OWLVIT_MODEL"

# Evaluate ASCR
echo "=== Evaluating ASCR ==="
"$PYTHON_BIN" scripts/evaluate_geneval_owlvit.py \
    "$GENEVAL_ASCR" \
    --outfile "${RESULTS_DIR}results_ascr.jsonl" \
    --model-path "$OWLVIT_MODEL"

# Print summary scores
echo "=== ShowO Baseline Scores ==="
python3 external/geneval/evaluation/summary_scores.py \
    "${RESULTS_DIR}results_baseline.jsonl"

echo "=== ASCR Scores ==="
python3 external/geneval/evaluation/summary_scores.py \
    "${RESULTS_DIR}results_ascr.jsonl"

echo "[geneval-eval] Done. Results in ${RESULTS_DIR}"

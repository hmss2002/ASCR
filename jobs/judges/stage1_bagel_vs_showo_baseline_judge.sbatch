#!/usr/bin/env bash
#SBATCH --job-name=bagel-vs-baseline-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV." >&2
  exit 2
fi
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

BAGEL_DIR="${BAGEL_DIR:?BAGEL_DIR must be set to the BAGEL bench run directory (e.g. outputs/bench3_bagel_YYYYMMDD_HHMMSS)}"
ASCR_RUN="${ASCR_RUN:?ASCR_RUN must be set to the ShowO/ASCR bench run root (e.g. outputs/bench3_showo_YYYYMMDD_HHMMSS)}"
CONFIG="configs/stage1/showo/stage1_showo_qwen35_9b_fullcap_parallel.yaml"
SUITE_OUT="$BAGEL_DIR/bagel_vs_showo_baseline_suite.json"
JUDGE_OUT="$BAGEL_DIR/qwen_pairwise_bagel_vs_showo_baseline.json"

echo "[judge] Step 1: pairing BAGEL images with ShowO baseline images"
"$PYTHON_BIN" scripts/benchmark/pair_bagel_vs_showo_baseline.py \
    --bagel-suite "$BAGEL_DIR/bagel_vs_ascr_suite.json" \
    --ascr-run-root "$ASCR_RUN" \
    --output "$SUITE_OUT"

echo "[judge] Step 2: pairwise Qwen judge (LEFT=ShowO baseline, RIGHT=BAGEL)"
"$PYTHON_BIN" scripts/judge/judge_showo_ascr_pairwise_qwen.py \
    "$SUITE_OUT" \
    --config "$CONFIG" \
    --output "$JUDGE_OUT" \
    --baseline-label "ShowO" \
    --ascr-label "BAGEL" \
    --no-image-labels

echo "[judge] Done: $JUDGE_OUT"

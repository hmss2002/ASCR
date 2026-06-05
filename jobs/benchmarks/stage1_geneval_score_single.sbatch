#!/usr/bin/env bash
#SBATCH --job-name=geneval-score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Score ONE GenEval-format image directory with the OWLViT objective evaluator
# (8 GPU shards). Each model is scored once; combine results across jobs with
# scripts/benchmark/build_geneval_3way_summary.py.
#
# Required env:
#   GENEVAL_IMAGES   Path to GenEval-style image dir (each prompt under "<idx>/samples/0.png")
#   GENEVAL_LABEL    Short label for the model (e.g. ShowO50, ASCR50, BAGEL)
# Optional env:
#   RESULTS_PREFIX   Output prefix path (default: $(dirname $GENEVAL_IMAGES)/score_$GENEVAL_LABEL)
#   OWLVIT_MODEL, DETR_MODEL, DETECTOR, NUM_SHARDS

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
source "$ASCR_ENV/bin/activate"
PYTHON_BIN="$ASCR_ENV/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

OWLVIT_MODEL=${OWLVIT_MODEL:-models/owlvit-base-patch32}
DETR_MODEL=${DETR_MODEL:-models/detr-resnet-50}
DETECTOR=${DETECTOR:-owlvit}
NUM_SHARDS=${NUM_SHARDS:-8}

: "${GENEVAL_IMAGES:?GENEVAL_IMAGES must point at a GenEval-style image dir}"
: "${GENEVAL_LABEL:?GENEVAL_LABEL is required (e.g. ShowO50, ASCR50, BAGEL)}"
[[ -f "$OWLVIT_MODEL/config.json" ]] || { echo "OWLViT missing at $OWLVIT_MODEL" >&2; exit 2; }

RESULTS_PREFIX="${RESULTS_PREFIX:-$(dirname "$GENEVAL_IMAGES")/score_${GENEVAL_LABEL}}"
mkdir -p "$(dirname "$RESULTS_PREFIX")"

echo "[geneval-score] dir   : $GENEVAL_IMAGES"
echo "[geneval-score] label : $GENEVAL_LABEL"
echo "[geneval-score] prefix: $RESULTS_PREFIX"
echo "[geneval-score] shards: $NUM_SHARDS"

PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$SHARD \
    "$PYTHON_BIN" scripts/judge/evaluate_geneval_owlvit.py \
      "$GENEVAL_IMAGES" \
      --outfile "${RESULTS_PREFIX}.shard${SHARD}.jsonl" \
      --model-path "$OWLVIT_MODEL" \
      --detector "$DETECTOR" \
      ${DETECTOR_EXTRA_ARGS:-} \
      --shard-id "$SHARD" \
      --num-shards "$NUM_SHARDS" \
      > "${RESULTS_PREFIX}.shard${SHARD}.log" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "  ERROR: shard $i failed (PID ${PIDS[$i]})" >&2
    FAILED=$((FAILED + 1))
  fi
done
[[ $FAILED -eq 0 ]] || { echo "ERROR: $FAILED shard(s) failed" >&2; exit 1; }

MERGED="${RESULTS_PREFIX}.jsonl"
> "$MERGED"
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  cat "${RESULTS_PREFIX}.shard${SHARD}.jsonl" >> "$MERGED"
done
echo "[geneval-score] merged → $MERGED ($(wc -l < "$MERGED") records)"

echo "=== ${GENEVAL_LABEL} Scores ==="
"$PYTHON_BIN" external/geneval/evaluation/summary_scores.py "$MERGED"

echo "[geneval-score] Done. Results at $MERGED"

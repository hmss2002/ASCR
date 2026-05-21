#!/usr/bin/env bash
#SBATCH --job-name=geneval-evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
source "$ASCR_ENV/bin/activate"
export PYTHON_BIN="$ASCR_ENV/bin/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

OWLVIT_MODEL=${OWLVIT_MODEL:-models/owlvit-base-patch32}
DETR_MODEL=${DETR_MODEL:-models/detr-resnet-50}
DETECTOR=${DETECTOR:-owlvit}
NUM_SHARDS=8

if [[ ! -f "$OWLVIT_MODEL/config.json" ]]; then
  echo "OWLViT model not found at $OWLVIT_MODEL" >&2
  echo "Run: python scripts/download_owlvit_model.py first (on login node)" >&2
  exit 2
fi

if [[ "$DETECTOR" == "detr" && ! -f "$DETR_MODEL/config.json" ]]; then
  echo "DETR model not found at $DETR_MODEL" >&2
  echo "Run: python scripts/download_detr_model.py first (on login node)" >&2
  exit 2
fi

if [[ -z "${GENEVAL_BASELINE:-}" || -z "${GENEVAL_ASCR:-}" ]]; then
  echo "ERROR: GENEVAL_BASELINE and GENEVAL_ASCR env vars must be set." >&2
  echo "  GENEVAL_BASELINE=path/to/geneval_baseline GENEVAL_ASCR=path/to/geneval_ascr sbatch ..." >&2
  exit 1
fi

# Derive results dir from parent of GENEVAL_BASELINE
RESULTS_DIR="$(dirname "$GENEVAL_BASELINE")/"

echo "[geneval-eval] Baseline dir:   $GENEVAL_BASELINE"
echo "[geneval-eval] ASCR dir:       $GENEVAL_ASCR"
echo "[geneval-eval] OWLViT model:   $OWLVIT_MODEL"
echo "[geneval-eval] Detector:      $DETECTOR"
[[ "$DETECTOR" == "detr" ]] && echo "[geneval-eval] DETR model:     $DETR_MODEL"
echo "[geneval-eval] Shards:         $NUM_SHARDS GPUs"

run_sharded_eval() {
  local IMAGEDIR="$1"
  local OUT_PREFIX="$2"
  local PIDS=()

  echo "[geneval-eval] Starting $NUM_SHARDS workers for $IMAGEDIR ..."
  for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$SHARD \
      "$PYTHON_BIN" scripts/evaluate_geneval_owlvit.py \
        "$IMAGEDIR" \
        --outfile "${OUT_PREFIX}.shard${SHARD}.jsonl" \
        --model-path "$OWLVIT_MODEL" \
        --detector "$DETECTOR" \
        ${DETECTOR_EXTRA_ARGS:-} \
        --shard-id "$SHARD" \
        --num-shards "$NUM_SHARDS" \
        > "${OUT_PREFIX}.shard${SHARD}.log" 2>&1 &
    PIDS+=($!)
    echo "  shard $SHARD → PID ${PIDS[-1]}"
  done

  # Wait for all shards and check exit codes
  local FAILED=0
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "  ERROR: shard $i failed (PID ${PIDS[$i]})" >&2
      FAILED=$((FAILED + 1))
    fi
  done
  if [[ $FAILED -gt 0 ]]; then
    echo "ERROR: $FAILED shard(s) failed for $IMAGEDIR" >&2
    exit 1
  fi

  # Merge shard results in order
  local MERGED="${OUT_PREFIX}.jsonl"
  > "$MERGED"
  for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    cat "${OUT_PREFIX}.shard${SHARD}.jsonl" >> "$MERGED"
  done
  echo "[geneval-eval] Merged → $MERGED ($(wc -l < "$MERGED") records)"
}

GENEVAL_BASELINE_LABEL=${GENEVAL_BASELINE_LABEL:-ShowO Baseline}
GENEVAL_ASCR_LABEL=${GENEVAL_ASCR_LABEL:-ASCR}

# === Evaluate Baseline ===
echo "=== Evaluating ${GENEVAL_BASELINE_LABEL} (8 GPU shards) ==="
run_sharded_eval "$GENEVAL_BASELINE" "${RESULTS_DIR}results_baseline"

# === Evaluate ASCR ===
echo "=== Evaluating ${GENEVAL_ASCR_LABEL} (8 GPU shards) ==="
run_sharded_eval "$GENEVAL_ASCR" "${RESULTS_DIR}results_ascr"

# === Print Summary Scores ===
echo "=== ${GENEVAL_BASELINE_LABEL} Scores ==="
python3 external/geneval/evaluation/summary_scores.py "${RESULTS_DIR}results_baseline.jsonl"

echo "=== ${GENEVAL_ASCR_LABEL} Scores ==="
python3 external/geneval/evaluation/summary_scores.py "${RESULTS_DIR}results_ascr.jsonl"

echo "[geneval-eval] Done. Results in ${RESULTS_DIR}"

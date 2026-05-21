#!/usr/bin/env bash
#SBATCH --job-name=hard64-bagel-3way-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# 8-GPU sharded Qwen3.5-9B pairwise judge for hard64 BAGEL vs {ShowO50, ASCR50}.
# Each pairing is split into 8 round-robin shards (one per GPU) and merged.
#
# Required env:
#   HARD64_RUN   New hard64 ASCR run root (must contain suite.json)
#   BAGEL_DIR    BAGEL hard64 run root (must contain bagel_vs_ascr_suite.json)
# Optional env:
#   CONFIG       Qwen judge config yaml (default: configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml)
#   NUM_SHARDS   Default 8

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

source "${ASCR_ENV:-.venv-qwen36}/bin/activate"
PY="${ASCR_ENV:-.venv-qwen36}/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

: "${HARD64_RUN:?HARD64_RUN must point to the new hard64 ASCR run root}"
: "${BAGEL_DIR:?BAGEL_DIR must point to the BAGEL hard64 run root}"
CONFIG="${CONFIG:-configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml}"
NUM_SHARDS=${NUM_SHARDS:-8}

HARD64_SUITE="$HARD64_RUN/suite.json"
BAGEL_SUITE="$BAGEL_DIR/bagel_vs_ascr_suite.json"
OUT_ROOT="$HARD64_RUN/bagel_3way"
mkdir -p "$OUT_ROOT"

echo "[3way] hard64 suite : $HARD64_SUITE"
echo "[3way] bagel suite  : $BAGEL_SUITE"
echo "[3way] num shards   : $NUM_SHARDS"
echo "[3way] output root  : $OUT_ROOT"

# Loop over each SIDE (ShowO50 vs BAGEL, ASCR50 vs BAGEL) and each DIRECTION
# (fwd = ShowO/ASCR LEFT BAGEL RIGHT, swap = BAGEL LEFT ShowO/ASCR RIGHT).
# Running both directions averages out Qwen position bias.
for SIDE in baseline ascr; do
  PEER_LABEL="$([[ $SIDE == baseline ]] && echo 'ShowO' || echo 'ASCR')"
  for DIR in fwd swap; do
    SUFFIX="${SIDE}_${DIR}"
    PAIRED="$OUT_ROOT/bagel_vs_${SUFFIX}_suite.json"
    SHARD_DIR="$OUT_ROOT/shards_bagel_vs_${SUFFIX}"
    mkdir -p "$SHARD_DIR"
    FINAL="$OUT_ROOT/qwen_pairwise_bagel_vs_${SUFFIX}.json"

    if [[ "$DIR" == "fwd" ]]; then
      SWAP_FLAG=""
      LEFT_LABEL="$PEER_LABEL"
      RIGHT_LABEL="BAGEL"
    else
      SWAP_FLAG="--swap"
      LEFT_LABEL="BAGEL"
      RIGHT_LABEL="$PEER_LABEL"
    fi

    echo "[3way] === Pairing LEFT=$LEFT_LABEL RIGHT=$RIGHT_LABEL ($SUFFIX) ==="
    "$PY" scripts/pair_bagel_vs_hard64_run.py \
        --bagel-suite "$BAGEL_SUITE" \
        --hard64-suite "$HARD64_SUITE" \
        --side "$SIDE" \
        $SWAP_FLAG \
        --output "$PAIRED"

    echo "[3way] === Sharded judge LEFT=$LEFT_LABEL RIGHT=$RIGHT_LABEL ($SUFFIX) ==="
    PIDS=()
    for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
      CUDA_VISIBLE_DEVICES=$SHARD \
        "$PY" scripts/judge_showo_ascr_pairwise_qwen.py \
            "$PAIRED" \
            --config "$CONFIG" \
            --output "$SHARD_DIR/shard${SHARD}.json" \
            --baseline-label "$LEFT_LABEL" \
            --ascr-label "$RIGHT_LABEL" \
            --no-image-labels \
            --shard-id "$SHARD" \
            --num-shards "$NUM_SHARDS" \
            > "$SHARD_DIR/shard${SHARD}.log" 2>&1 &
      PIDS+=($!)
      echo "  shard $SHARD → PID ${PIDS[-1]}"
    done

    FAILED=0
    for i in "${!PIDS[@]}"; do
      if ! wait "${PIDS[$i]}"; then
        echo "  ERROR: shard $i failed (PID ${PIDS[$i]}); see $SHARD_DIR/shard${i}.log" >&2
        FAILED=$((FAILED + 1))
      fi
    done
    [[ $FAILED -eq 0 ]] || { echo "ERROR: $FAILED shard(s) failed for $SUFFIX" >&2; exit 1; }

    echo "[3way] === Merging shards → $FINAL ==="
    "$PY" scripts/merge_judge_shards.py \
        $SHARD_DIR/shard*.json \
        --output "$FINAL"
  done
done

echo "[3way] Done. Outputs under $OUT_ROOT/"

#!/usr/bin/env bash
#SBATCH --job-name=hard64-showo-ascr-swap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# 8-GPU sharded Qwen3.5-9B pairwise judge for hard64 ShowO vs ASCR with SWAPPED
# image positions. Forward direction (ASCR on RIGHT) is already in
# $RUN_ROOT/qwen_pairwise_judge.json. This run puts ASCR on LEFT to debias
# Qwen RIGHT-side preference.
#
# Required env:
#   RUN_ROOT   hard64 ASCR run root (must contain suite.json)

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

source "${ASCR_ENV:-.venv-qwen36}/bin/activate"
PY="${ASCR_ENV:-.venv-qwen36}/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

: "${RUN_ROOT:?RUN_ROOT must point to the hard64 ASCR run root}"
CONFIG="${CONFIG:-configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml}"
NUM_SHARDS=${NUM_SHARDS:-8}

SUITE_IN="$RUN_ROOT/suite.json"
OUT_ROOT="$RUN_ROOT/diagnostics_swap"
SHARD_DIR="$OUT_ROOT/shards"
SWAPPED_SUITE="$OUT_ROOT/suite_swapped.json"
FINAL="$OUT_ROOT/qwen_pairwise_SWAPPED.json"
mkdir -p "$SHARD_DIR"

echo "[swap-diag] suite_in   : $SUITE_IN"
echo "[swap-diag] num shards : $NUM_SHARDS"
echo "[swap-diag] output     : $FINAL"

# 1) Build swapped suite (flip baseline_image <-> ascr_final_image)
"$PY" - "$SUITE_IN" "$SWAPPED_SUITE" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src))
results = data.get("results", [])
for r in results:
    base = r.get("baseline_image")
    ascr = (r.get("accepted_ascr_image") or r.get("ascr_final_image")
            or (r.get("ascr_summary") or {}).get("final_decoded_image"))
    if base is None or ascr is None:
        continue
    # Swap: new baseline_image is original ascr; new ascr_final_image is original baseline.
    r["baseline_image"] = ascr
    r["ascr_final_image"] = base
    # Remove fallback fields that would override ascr_final_image
    r.pop("accepted_ascr_image", None)
    if "ascr_summary" in r:
        r["ascr_summary"].pop("final_decoded_image", None)
json.dump(data, open(dst, "w"), indent=2)
print(f"swapped suite written: {dst}")
PYEOF

# 2) Sharded judge with labels reflecting the swap (LEFT=ASCR, RIGHT=ShowO)
PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
  SHARD_OUT="$SHARD_DIR/shard${SHARD}.json"
  SHARD_LOG="$SHARD_DIR/shard${SHARD}.log"
  CUDA_VISIBLE_DEVICES=$SHARD \
    "$PY" scripts/judge_showo_ascr_pairwise_qwen.py "$SWAPPED_SUITE" \
        --config "$CONFIG" \
        --output "$SHARD_OUT" \
        --num-shards "$NUM_SHARDS" --shard-id "$SHARD" \
        --baseline-label ASCR --ascr-label ShowO \
        --no-image-labels \
        > "$SHARD_LOG" 2>&1 &
  PIDS+=($!)
  echo "  shard $SHARD -> PID ${PIDS[-1]}"
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "ERROR: shard $i failed" >&2
    FAILED=$((FAILED+1))
  fi
done
[[ $FAILED -eq 0 ]] || exit 1

# 3) Merge
"$PY" scripts/merge_judge_shards.py \
    $SHARD_DIR/shard*.json \
    --output "$FINAL"
echo "[swap-diag] DONE -> $FINAL"

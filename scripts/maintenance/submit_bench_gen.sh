#!/usr/bin/env bash
# submit_bench_gen.sh
# Splits bench3_combined.txt and submits ShowO+ASCR generation jobs across nodes.
# Usage: bash scripts/maintenance/submit_bench_gen.sh [N_NODES]
#
# Env vars:
#   PROMPTS_FILE   - combined prompts file (default: configs/benchmarks/prompts/bench3_combined.txt)
#   N_NODES        - number of shards/nodes (default: 8)
#   TS             - timestamp tag (auto)

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

PROMPTS_FILE="${PROMPTS_FILE:-configs/benchmarks/prompts/bench3_combined.txt}"
N_NODES="${1:-${N_NODES:-8}}"
TS=$(date +%Y%m%d_%H%M%S)
ARRAY_RUN_ROOT="outputs/bench3_showo_${TS}"
mkdir -p "$ARRAY_RUN_ROOT"

echo "========================================================"
echo " Bench3 ShowO+ASCR generation"
echo " Prompts file : $PROMPTS_FILE"
echo " N nodes      : $N_NODES"
echo " Run root     : $ARRAY_RUN_ROOT"
echo "========================================================"

# Split prompts
SPLIT_DIR="$ARRAY_RUN_ROOT/prompt_splits"
mkdir -p "$SPLIT_DIR"
TOTAL=$(wc -l < "$PROMPTS_FILE")
CHUNK=$(( (TOTAL + N_NODES - 1) / N_NODES ))
for ((i=0; i<N_NODES; i++)); do
  START=$(( i * CHUNK + 1 ))
  END=$(( (i + 1) * CHUNK ))
  [[ $END -gt $TOTAL ]] && END=$TOTAL
  SLICE="$SPLIT_DIR/slice_${i}.txt"
  sed -n "${START},${END}p" "$PROMPTS_FILE" > "$SLICE"
  # Record line offsets for image_map construction
  echo "$i $START $END" >> "$SPLIT_DIR/slice_offsets.txt"
  echo "  slice $i: lines ${START}-${END} ($(wc -l < "$SLICE") prompts)"
done

# Save metadata for downstream build_bench_image_map.py
cat > "$ARRAY_RUN_ROOT/bench_meta.json" <<JSON
{
  "prompts_file": "$PROMPTS_FILE",
  "bench3_index": "configs/benchmarks/data/bench3_index.json",
  "n_nodes": $N_NODES,
  "timestamp": "$TS"
}
JSON

# Submit gen jobs — first 3 to gpu, rest to gpu_shared
GEN_JIDS=()
for ((i=0; i<N_NODES; i++)); do
  SLICE="$SPLIT_DIR/slice_${i}.txt"
  PART="gpu_shared"
  [[ $i -lt 3 ]] && PART="gpu"
  JID=$(sbatch --parsable \
    --partition="$PART" \
    --export=ALL,SHARD_PROMPTS_FILE="$SLICE",ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
    jobs/benchmarks/geneval_gen_shard.sbatch)
  GEN_JIDS+=("$JID")
  echo "  submitted job $JID (partition=$PART, slice=$i)"
done

# Submit merge job (depends on all gen jobs finishing)
DEP=$(IFS=:; echo "afterok:${GEN_JIDS[*]}")
MERGE_JID=$(sbatch --parsable \
  --dependency="$DEP" \
  --export=ALL,ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
  jobs/benchmarks/geneval_merge_eval.sbatch)
echo "  submitted merge job $MERGE_JID (dep: $DEP)"

echo ""
echo "========================================================"
echo " Gen jobs   : ${GEN_JIDS[*]}"
echo " Merge job  : $MERGE_JID"
echo " Run root   : $ARRAY_RUN_ROOT"
echo ""
echo " After merge completes, run:"
echo "   python scripts/benchmark/build_bench_image_map.py \\"
echo "     --run-root $ARRAY_RUN_ROOT \\"
echo "     --bench3-index configs/benchmarks/data/bench3_index.json"
echo "========================================================"

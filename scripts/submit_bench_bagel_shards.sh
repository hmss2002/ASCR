#!/usr/bin/env bash
# submit_bench_bagel_shards.sh
# Submits N_SHARDS BAGEL shard jobs to generate all bench3 images (3725 prompts).
# Usage: bash scripts/submit_bench_bagel_shards.sh [N_SHARDS]
#
# Env vars:
#   PROMPT_FILE  - combined prompts file (default: configs/prompts/bench3_combined.txt)
#   N_SHARDS     - number of parallel shard jobs (default: 8)

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

PROMPT_FILE="${PROMPT_FILE:-configs/prompts/bench3_combined.txt}"
N_SHARDS="${1:-${N_SHARDS:-8}}"
TS=$(date +%Y%m%d_%H%M%S)
BENCH_RUN_ROOT="outputs/bench3_bagel_${TS}"
mkdir -p "$BENCH_RUN_ROOT"

TOTAL=$(wc -l < "$PROMPT_FILE")
CHUNK=$(( (TOTAL + N_SHARDS - 1) / N_SHARDS ))

echo "========================================================"
echo " Bench3 BAGEL generation"
echo " Prompts    : $TOTAL ($PROMPT_FILE)"
echo " Shards     : $N_SHARDS (~$CHUNK prompts each)"
echo " Run root   : $BENCH_RUN_ROOT"
echo "========================================================"

# Save metadata
cat > "$BENCH_RUN_ROOT/bench_meta.json" <<JSON
{
  "prompt_file": "$PROMPT_FILE",
  "bench3_index": "configs/benchmark_data/bench3_index.json",
  "n_shards": $N_SHARDS,
  "timestamp": "$TS",
  "total_prompts": $TOTAL
}
JSON

JIDS=()
for ((i=0; i<N_SHARDS; i++)); do
  OFFSET=$(( i * CHUNK ))
  # Limit for last shard: don't exceed total
  LIM=$CHUNK
  [[ $(( OFFSET + LIM )) -gt $TOTAL ]] && LIM=$(( TOTAL - OFFSET ))
  JID=$(sbatch --parsable \
    --export=ALL,BENCH_RUN_ROOT="$BENCH_RUN_ROOT",PROMPT_FILE="$PROMPT_FILE",PROMPT_OFFSET="$OFFSET",PROMPT_LIMIT="$LIM" \
    jobs/bench_bagel_gen_shard.sbatch)
  JIDS+=("$JID")
  echo "  submitted shard $i: job $JID (offset=$OFFSET limit=$LIM)"
done

echo ""
echo "========================================================"
echo " BAGEL shard jobs: ${JIDS[*]}"
echo " Run root        : $BENCH_RUN_ROOT"
echo ""
echo " After all shards complete, run:"
echo "   python scripts/build_bench_image_map.py \\"
echo "     --bagel-run-root $BENCH_RUN_ROOT \\"
echo "     --bench3-index configs/benchmark_data/bench3_index.json"
echo "========================================================"

#!/usr/bin/env bash
# Re-submit BAGEL bench3 shards that missed due to QOS limit.
# Required env vars:
#   BENCH_RUN_ROOT  - run root from the original submit_bench_bagel_shards.sh run
#   PROMPT_OFFSET_N, PROMPT_LIMIT_N  - offset+limit for each shard N to resubmit
# Example:
#   BENCH_RUN_ROOT=outputs/bench3_bagel_YYYYMMDD \
#   SLICES="5 6 7" \
#   bash scripts/maintenance/submit_bench_bagel_remaining.sh
set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR

BENCH_RUN_ROOT="${BENCH_RUN_ROOT:?Must set BENCH_RUN_ROOT to the bench3 BAGEL run directory}"
PROMPT_FILE="${PROMPT_FILE:-configs/benchmarks/prompts/bench3_combined.txt}"
SLICES="${SLICES:-5 6 7}"

echo "Submitting remaining BAGEL shards to $BENCH_RUN_ROOT..."
echo "Slices: $SLICES"
echo "NOTE: Set PROMPT_OFFSET_<n> and PROMPT_LIMIT_<n> for each slice before running."

JOB_IDS=()
for i in $SLICES; do
  OFFSET_VAR="PROMPT_OFFSET_${i}"
  LIMIT_VAR="PROMPT_LIMIT_${i}"
  OFFSET="${!OFFSET_VAR:?Must set ${OFFSET_VAR}}"
  LIMIT="${!LIMIT_VAR:?Must set ${LIMIT_VAR}}"
  JID=$(sbatch --parsable \
    --export=ALL,BENCH_RUN_ROOT="$BENCH_RUN_ROOT",PROMPT_FILE="$PROMPT_FILE",PROMPT_OFFSET="$OFFSET",PROMPT_LIMIT="$LIMIT" \
    jobs/benchmarks/bench_bagel_gen_shard.sbatch)
  JOB_IDS+=("$JID")
  echo "  shard $i: job $JID (offset=$OFFSET limit=$LIMIT)"
done

echo "Done! Jobs: ${JOB_IDS[*]}"

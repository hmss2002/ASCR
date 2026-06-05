#!/usr/bin/env bash
# Re-submit ShowO/ASCR bench3 shards that missed due to QOS limit.
# Required env vars:
#   ARRAY_RUN_ROOT     - run root from the original submit_bench_gen.sh run
#   COMPLETED_JIDS     - colon-separated IDs of already-completed shard jobs (for merge dependency)
#   SLICES             - space-separated shard indices to resubmit (e.g. "5 6 7")
set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR

ARRAY_RUN_ROOT="${ARRAY_RUN_ROOT:?Must set ARRAY_RUN_ROOT to the bench3 ShowO run directory}"
COMPLETED_JIDS="${COMPLETED_JIDS:?Must set COMPLETED_JIDS to colon-separated IDs of completed shard jobs}"
SLICES="${SLICES:-5 6 7}"
SPLIT_DIR="$ARRAY_RUN_ROOT/prompt_splits"

NEW_JIDS=()
for i in $SLICES; do
  JID=$(sbatch --parsable --partition=gpu_shared \
    --export=ALL,SHARD_PROMPTS_FILE="$SPLIT_DIR/slice_${i}.txt",ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
    jobs/benchmarks/geneval_gen_shard.sbatch)
  NEW_JIDS+=("$JID")
  echo "  slice $i: job $JID"
done

DEP_IDS=$(IFS=:; echo "${COMPLETED_JIDS}:${NEW_JIDS[*]}")
MERGE_JID=$(sbatch --parsable --dependency="afterok:${DEP_IDS}" \
  --export=ALL,ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
  jobs/benchmarks/geneval_merge_eval.sbatch)
echo "  merge: job $MERGE_JID (dep: afterok:${DEP_IDS})"
echo "Done! Run root: $ARRAY_RUN_ROOT"

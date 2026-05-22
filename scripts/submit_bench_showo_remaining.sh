#!/usr/bin/env bash
set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
ARRAY_RUN_ROOT="outputs/bench3_showo_20260522_210258"
SPLIT_DIR="$ARRAY_RUN_ROOT/prompt_splits"

# Submit slices 5-7 (missed due to QOS limit)
JID5=$(sbatch --parsable --partition=gpu_shared \
  --export=ALL,SHARD_PROMPTS_FILE="$SPLIT_DIR/slice_5.txt",ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
  jobs/geneval_gen_shard.sbatch)
echo "  slice 5: job $JID5"

JID6=$(sbatch --parsable --partition=gpu_shared \
  --export=ALL,SHARD_PROMPTS_FILE="$SPLIT_DIR/slice_6.txt",ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
  jobs/geneval_gen_shard.sbatch)
echo "  slice 6: job $JID6"

JID7=$(sbatch --parsable --partition=gpu_shared \
  --export=ALL,SHARD_PROMPTS_FILE="$SPLIT_DIR/slice_7.txt",ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
  jobs/geneval_gen_shard.sbatch)
echo "  slice 7: job $JID7"

# Merge depends on 68878-68882 + new 3 jobs
DEP="afterok:68878:68879:68880:68881:68882:$JID5:$JID6:$JID7"
MERGE_JID=$(sbatch --parsable --dependency="$DEP" \
  --export=ALL,ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
  jobs/geneval_merge_eval.sbatch)
echo "  merge: job $MERGE_JID (dep: $DEP)"
echo "Done! Run root: $ARRAY_RUN_ROOT"

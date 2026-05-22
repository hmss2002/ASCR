#!/usr/bin/env bash
# Submit remaining BAGEL bench3 shards (5, 6, 7) that couldn't submit due to QOS limit.
set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR

BENCH_RUN_ROOT="outputs/bench3_bagel_20260522_193546"
PROMPT_FILE="configs/prompts/bench3_combined.txt"

echo "Submitting remaining BAGEL shards to $BENCH_RUN_ROOT..."

# shard 5: offset=2330 limit=466
JID5=$(sbatch --parsable \
  --export=ALL,BENCH_RUN_ROOT="$BENCH_RUN_ROOT",PROMPT_FILE="$PROMPT_FILE",PROMPT_OFFSET=2330,PROMPT_LIMIT=466 \
  jobs/bench_bagel_gen_shard.sbatch)
echo "  shard 5: job $JID5 (offset=2330 limit=466)"

# shard 6: offset=2796 limit=466
JID6=$(sbatch --parsable \
  --export=ALL,BENCH_RUN_ROOT="$BENCH_RUN_ROOT",PROMPT_FILE="$PROMPT_FILE",PROMPT_OFFSET=2796,PROMPT_LIMIT=466 \
  jobs/bench_bagel_gen_shard.sbatch)
echo "  shard 6: job $JID6 (offset=2796 limit=466)"

# shard 7: offset=3262 limit=463
JID7=$(sbatch --parsable \
  --export=ALL,BENCH_RUN_ROOT="$BENCH_RUN_ROOT",PROMPT_FILE="$PROMPT_FILE",PROMPT_OFFSET=3262,PROMPT_LIMIT=463 \
  jobs/bench_bagel_gen_shard.sbatch)
echo "  shard 7: job $JID7 (offset=3262 limit=463)"

echo "Done! Jobs: $JID5 $JID6 $JID7"

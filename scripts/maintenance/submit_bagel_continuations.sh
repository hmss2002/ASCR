#!/usr/bin/env bash
# submit_bagel_continuations.sh
# Submits continuation jobs for BAGEL bench3 shards that timed out.
# Uses --dependency=afternotok:<orig_jid> (works even after orig job is FAILED).
# Uses SHARD_OUT_OVERRIDE + SKIP_EXISTING=1 to resume into the same directory.
#
# Usage: bash scripts/maintenance/submit_bagel_continuations.sh
# Can be run repeatedly; jobs that are already queued won't be re-submitted.

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-outputs/bench3_bagel_20260522_193546}"
PROMPT_FILE="${PROMPT_FILE:-configs/benchmarks/prompts/bench3_combined.txt}"

# Original shard job IDs and their assignments
declare -A OFFSET=([68869]=0 [68870]=466 [68871]=932 [68872]=1398 [68873]=1864 [68875]=2330 [68876]=2796 [68877]=3262)
declare -A LIMIT=([68869]=466 [68870]=466 [68871]=466 [68872]=466 [68873]=466 [68875]=466 [68876]=466 [68877]=463)

echo "================================================"
echo " BAGEL bench3 continuation jobs"
echo " Run root: $BENCH_RUN_ROOT"
echo "================================================"

submitted=0
failed=0
for orig_jid in 68869 68870 68871 68872 68873 68875 68876 68877; do
  offset="${OFFSET[$orig_jid]}"
  limit="${LIMIT[$orig_jid]}"
  shard_dir="$BENCH_RUN_ROOT/shard_${orig_jid}"
  done_count=$(find "$shard_dir/images" -name "*.png" 2>/dev/null | wc -l)

  if [[ "$done_count" -ge "$limit" ]]; then
    echo "  shard $orig_jid: already complete ($done_count/$limit images) — skipping"
    continue
  fi

  state=$(sacct -j "$orig_jid" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  if [[ "$state" == "RUNNING" ]]; then
    echo "  shard $orig_jid: still RUNNING ($done_count/$limit done) — will queue with afternotok dep"
  elif [[ "$state" == "COMPLETED" ]]; then
    echo "  shard $orig_jid: COMPLETED — skipping"
    continue
  else
    echo "  shard $orig_jid: state=$state ($done_count/$limit done) — queueing continuation"
  fi

  if new_jid=$(sbatch --parsable \
    --dependency="afternotok:${orig_jid}" \
    --partition=gpu_shared \
    --export=ALL,BENCH_RUN_ROOT="$BENCH_RUN_ROOT",PROMPT_FILE="$PROMPT_FILE",PROMPT_OFFSET="$offset",PROMPT_LIMIT="$limit",SHARD_OUT_OVERRIDE="$shard_dir",SKIP_EXISTING=1 \
    jobs/benchmarks/bench_bagel_gen_shard.sbatch 2>&1); then
    echo "    → submitted continuation job $new_jid"
    submitted=$((submitted + 1))
  else
    echo "    → FAILED to submit: $new_jid"
    failed=$((failed + 1))
  fi
done

echo "================================================"
echo " Submitted: $submitted  Failed: $failed"
if [[ "$failed" -gt 0 ]]; then
  echo " Note: QOS limit hit. Run this script again once jobs free up."
fi
echo "================================================"

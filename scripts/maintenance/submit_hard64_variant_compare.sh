#!/usr/bin/env bash
set -euo pipefail

# Submit the Hard64 variant-comparison generation jobs.
# For each ARM in {coarse,direct} and each node 0..NODES_PER_ARM-1, submit one
# 8-GPU node job. Default 2 arms x 3 nodes x 8 = 48 GPUs (<= 56 cap, <= 8 jobs).
#
# Each arm shares a single RUN_ROOT across its nodes so shards stitch together.
#
# Env:
#   NODES_PER_ARM   nodes per arm (default 3)
#   ARMS            space/comma list of arms (default "coarse direct")
#   PARTITION       slurm partition (default gpu)
#   PROMPT_LIMIT    prompt cap (default 64)
#   DRY_RUN         1 = print sbatch commands without submitting

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$ROOT"

NODES_PER_ARM=${NODES_PER_ARM:-3}
ARMS=${ARMS:-"coarse direct"}
ARMS=${ARMS//,/ }
PARTITION=${PARTITION:-gpu}
PROMPT_LIMIT=${PROMPT_LIMIT:-64}
PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-jobs/stage1/variants/stage1_hard64_variant_gen_8gpu.sbatch}
LOCAL_WORKERS=${LOCAL_WORKERS:-8}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
DRY_RUN=${DRY_RUN:-0}

mkdir -p logs outputs

MANIFEST="outputs/hard64_variant_submit_${STAMP}.txt"
: > "$MANIFEST"

for arm in $ARMS; do
  run_root="outputs/benchmarks_hard64_variant_${arm}_${STAMP}"
  global_workers=$((NODES_PER_ARM * LOCAL_WORKERS))
  echo "# arm=$arm run_root=$run_root global_workers=$global_workers" | tee -a "$MANIFEST"
  for ((node = 0; node < NODES_PER_ARM; node++)); do
    global_offset=$((node * LOCAL_WORKERS))
    cmd=(sbatch
      --partition="$PARTITION"
      --job-name="ascr-h64-${arm}-n${node}"
      --export=ALL,ARM="$arm",NODE_INDEX="$node",NODE_COUNT="$NODES_PER_ARM",GLOBAL_WORKERS="$global_workers",GLOBAL_OFFSET="$global_offset",RUN_ROOT="$run_root",PROMPT_LIMIT="$PROMPT_LIMIT",PROMPTS_FILE="$PROMPTS_FILE",LOCAL_WORKERS="$LOCAL_WORKERS"
      "$SBATCH_SCRIPT")
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] ${cmd[*]}" | tee -a "$MANIFEST"
    else
      out=$("${cmd[@]}")
      echo "$out (arm=$arm node=$node offset=$global_offset run_root=$run_root)" | tee -a "$MANIFEST"
    fi
  done
done

echo "[submit] manifest: $MANIFEST"
echo "[submit] after jobs finish, collect with: scripts/run/run_hard64_variant_gemini.sh"

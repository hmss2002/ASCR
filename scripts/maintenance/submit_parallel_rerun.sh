#!/usr/bin/env bash
# submit_parallel_rerun.sh
# Splits geneval/hard64 prompts and submits all jobs in parallel across
# available GPU nodes (gpu + gpu_shared partitions), then chains a
# merge+evaluate job.
set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

# ── config ──────────────────────────────────────────────────────────────────
GENEVAL_PROMPTS="configs/benchmarks/prompts/geneval_553.txt"
HARD64_PROMPTS="configs/benchmarks/prompts/t2i_compbench_hard64.txt"
N_GENEVAL_NODES=${N_GENEVAL_NODES:-9}   # 3 gpu-idle + 6 gpu_shared-idle
TS=$(date +%Y%m%d_%H%M%S)
ARRAY_RUN_ROOT="outputs/geneval_parallel_${TS}"
HARD64_RUN_ROOT="outputs/hard64_parallel_${TS}"
mkdir -p "$ARRAY_RUN_ROOT" "$HARD64_RUN_ROOT"

echo "============================================================"
echo " GenEval run root : $ARRAY_RUN_ROOT"
echo " Hard64  run root : $HARD64_RUN_ROOT"
echo " GenEval nodes    : $N_GENEVAL_NODES"
echo "============================================================"

# ── split geneval prompts ────────────────────────────────────────────────────
SPLIT_DIR="$ARRAY_RUN_ROOT/prompt_splits"
mkdir -p "$SPLIT_DIR"
TOTAL_GENEVAL=$(wc -l < "$GENEVAL_PROMPTS")
CHUNK=$(( (TOTAL_GENEVAL + N_GENEVAL_NODES - 1) / N_GENEVAL_NODES ))
for ((i=0; i<N_GENEVAL_NODES; i++)); do
  START=$(( i * CHUNK + 1 ))
  END=$(( (i + 1) * CHUNK ))
  [[ $END -gt $TOTAL_GENEVAL ]] && END=$TOTAL_GENEVAL
  SLICE="$SPLIT_DIR/slice_${i}.txt"
  sed -n "${START},${END}p" "$GENEVAL_PROMPTS" > "$SLICE"
  echo "  slice $i: lines ${START}-${END} ($(wc -l < "$SLICE") prompts) → $SLICE"
done

# ── submit geneval generation jobs ───────────────────────────────────────────
# First 3 jobs → gpu partition (3 idle nodes), rest → gpu_shared
GEN_JIDS=()
for ((i=0; i<N_GENEVAL_NODES; i++)); do
  SLICE="$SPLIT_DIR/slice_${i}.txt"
  if [[ $i -lt 3 ]]; then
    PART="gpu"
  else
    PART="gpu_shared"
  fi
  JID=$(sbatch --parsable \
    --partition="$PART" \
    --export=ALL,SHARD_PROMPTS_FILE="$SLICE",ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
    jobs/benchmarks/geneval_gen_shard.sbatch)
  GEN_JIDS+=("$JID")
  echo "  [geneval] submitted job $JID (partition=$PART, slice=$i)"
done

# ── submit merge+eval job (depends on all gen jobs) ─────────────────────────
DEP=$(IFS=:; echo "afterok:${GEN_JIDS[*]}")
MERGE_JID=$(sbatch --parsable \
  --dependency="$DEP" \
  --export=ALL,ARRAY_RUN_ROOT="$ARRAY_RUN_ROOT" \
  jobs/benchmarks/geneval_merge_eval.sbatch)
echo "  [merge-eval] submitted job $MERGE_JID (dep: $DEP)"

# ── submit hard64 job (single node, gpu_shared) ──────────────────────────────
H64_JID=$(sbatch --parsable \
  --partition=gpu_shared \
  --export=ALL,RUN_ROOT="$HARD64_RUN_ROOT",PROMPTS_FILE="$HARD64_PROMPTS",PROMPT_LIMIT=64 \
  jobs/stage1/showo/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch)
echo "  [hard64]   submitted job $H64_JID (partition=gpu_shared)"

echo ""
echo "============================================================"
echo " All jobs submitted!"
echo " GenEval gen jobs : ${GEN_JIDS[*]}"
echo " GenEval merge    : $MERGE_JID  (starts after all gen jobs finish)"
echo " Hard64 job       : $H64_JID"
echo ""
echo " Monitor: squeue -u \$USER"
echo " GenEval logs    : $ARRAY_RUN_ROOT"
echo " Hard64 logs     : $HARD64_RUN_ROOT"
echo "============================================================"

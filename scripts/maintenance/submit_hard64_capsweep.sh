#!/usr/bin/env bash
set -euo pipefail

# Submit the Phase-6 budget-matched cap-sweep generation jobs.
# For each reopened-token budget B in {64,256,512} and each arm in {direct,coarse},
# submit ONE single-node GPU job that regenerates Hard64 with the matched config:
#   direct b{B} : max_selected_cells=B   (token units),  dilation=0
#   coarse b{B} : max_selected_cells=B/64 (4x4-cell units), dilation=0
# Both arms share the same deterministic baseline (seed 1234), so pass-rate
# differences isolate the reopen strategy at an equal reopened-token budget.
#
# Env:
#   BUDGETS         space/comma list of B values (default "64 256 512")
#   ARMS            space/comma list of arms     (default "direct coarse")
#   PARTITION       slurm partition              (default gpu_shared)
#   GPUS_PER_RUN    GPUs (=resident workers) per run (default 4)
#   PROMPT_LIMIT    prompt cap                   (default 64)
#   STAMP           shared timestamp for run roots (default: now)
#   DRY_RUN         1 = print sbatch commands without submitting

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$ROOT"

BUDGETS=${BUDGETS:-"64 256 512"};  BUDGETS=${BUDGETS//,/ }
ARMS=${ARMS:-"direct coarse"};     ARMS=${ARMS//,/ }
PARTITION=${PARTITION:-gpu_shared}
GPUS_PER_RUN=${GPUS_PER_RUN:-4}
PROMPT_LIMIT=${PROMPT_LIMIT:-64}
PROMPTS_FILE=${PROMPTS_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-jobs/stage1/variants/stage1_hard64_variant_gen_8gpu.sbatch}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
DRY_RUN=${DRY_RUN:-0}

mkdir -p logs outputs
MANIFEST="outputs/hard64_capsweep_submit_${STAMP}.txt"
: > "$MANIFEST"
echo "# cap-sweep stamp=$STAMP partition=$PARTITION gpus_per_run=$GPUS_PER_RUN" | tee -a "$MANIFEST"

for B in $BUDGETS; do
  for arm in $ARMS; do
    if [[ "$arm" == "direct" ]]; then
      cfg="configs/stage1/showo/stage1_showo_qwen35_9b_direct_token_b${B}.yaml"
      coarse_cfg="configs/stage1/showo/stage1_showo_qwen35_9b_coarse_b${B}.yaml"
    else
      cfg="configs/stage1/showo/stage1_showo_qwen35_9b_direct_token_b${B}.yaml"
      coarse_cfg="configs/stage1/showo/stage1_showo_qwen35_9b_coarse_b${B}.yaml"
    fi
    if [[ ! -f "$cfg" || ! -f "$coarse_cfg" ]]; then
      echo "ERROR: missing config for arm=$arm B=$B ($cfg / $coarse_cfg)" >&2
      exit 2
    fi
    run_root="outputs/benchmarks_hard64_capsweep_${arm}_b${B}_${STAMP}"
    cmd=(sbatch
      --partition="$PARTITION"
      --gres="gpu:${GPUS_PER_RUN}"
      --job-name="ascr-cap-${arm}-b${B}"
      --export=ALL,ARM="$arm",NODE_INDEX=0,NODE_COUNT=1,LOCAL_WORKERS="$GPUS_PER_RUN",GLOBAL_WORKERS="$GPUS_PER_RUN",GLOBAL_OFFSET=0,RUN_ROOT="$run_root",CONFIG="$cfg",COARSE_CONFIG="$coarse_cfg",PROMPT_LIMIT="$PROMPT_LIMIT",PROMPTS_FILE="$PROMPTS_FILE"
      "$SBATCH_SCRIPT")
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] ${cmd[*]}" | tee -a "$MANIFEST"
    else
      out=$("${cmd[@]}")
      echo "$out (arm=$arm B=$B run_root=$run_root cfg=$cfg coarse_cfg=$coarse_cfg)" | tee -a "$MANIFEST"
    fi
  done
done

echo "[submit] manifest: $MANIFEST"
echo "[submit] STAMP=$STAMP  (reuse for the judging step)"

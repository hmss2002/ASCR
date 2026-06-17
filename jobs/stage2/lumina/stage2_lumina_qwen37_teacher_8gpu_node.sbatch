#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage2-lumina-8gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

CONFIG=${CONFIG:-configs/stage2/lumina/stage2_lumina_qwen37_teacher_hq.yaml}
OUT_ROOT=${OUT_ROOT:-outputs/stage2_lumina_qwen37_teacher_hq}
PROMPT_FILE=${PROMPT_FILE:-}
ASCR_ENV=${ASCR_ENV:-.venv-lumina}
WORKERS_PER_NODE=${WORKERS_PER_NODE:-8}
NODE_INDEX=${NODE_INDEX:-0}
NODE_COUNT=${NODE_COUNT:-1}

if [[ -z "${OFOX_API_KEY:-}" ]]; then
  echo "OFOX_API_KEY is required for Stage 2 teacher runs." >&2
  exit 2
fi
if [[ -z "$PROMPT_FILE" || ! -f "$PROMPT_FILE" ]]; then
  echo "PROMPT_FILE must point to an existing prompt list." >&2
  exit 2
fi
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV" >&2
  exit 2
fi

# Current Stage 2 code path is safest as 8 independent single-GPU workers on one node.
# If native multi-GPU Lumina inference becomes stable later, this script can be replaced
# by a real tensor-parallel or DDP launcher.
source "$ASCR_ENV/bin/activate"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
mapfile -t PROMPTS < <(grep -v '^[[:space:]]*$' "$PROMPT_FILE")

worker_pids=()
for ((worker=0; worker<WORKERS_PER_NODE; worker++)); do
  (
    export CUDA_VISIBLE_DEVICES="$worker"
    worker_dir="$OUT_ROOT/node_${NODE_INDEX}_worker_${worker}"
    mkdir -p "$worker_dir"
    for ((index=worker + NODE_INDEX * WORKERS_PER_NODE; index<${#PROMPTS[@]}; index+=WORKERS_PER_NODE * NODE_COUNT)); do
      prompt_text="${PROMPTS[$index]}"
      prompt_dir="$worker_dir/prompt_$(printf '%05d' "$index")"
      mkdir -p "$prompt_dir"
      python -m ascr.cli.run_stage1 \
        --config "$CONFIG" \
        --output-dir "$prompt_dir" \
        --prompt "$prompt_text"
    done
  ) > "logs/${SLURM_JOB_NAME:-stage2}-${SLURM_JOB_ID:-local}-w${worker}.log" 2>&1 &
  worker_pids+=($!)
done

fail=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || fail=1
done
exit "$fail"

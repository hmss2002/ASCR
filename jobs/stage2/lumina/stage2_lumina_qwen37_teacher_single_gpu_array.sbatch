#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage2-lumina-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# NOTE: submit with an explicit array range, e.g.
#   sbatch --array=0-255%56 jobs/stage2/lumina/stage2_lumina_qwen37_teacher_single_gpu_array.sbatch

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

CONFIG=${CONFIG:-configs/stage2/lumina/stage2_lumina_qwen37_teacher_hq.yaml}
OUT_ROOT=${OUT_ROOT:-outputs/stage2_lumina_qwen37_teacher_hq}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-1}
ASCR_ENV=${ASCR_ENV:-.venv-lumina}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

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

source "$ASCR_ENV/bin/activate"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

mapfile -t PROMPTS < <(grep -v '^[[:space:]]*$' "$PROMPT_FILE")
START=$((TASK_ID * PROMPTS_PER_TASK))
END=$((START + PROMPTS_PER_TASK))
SHARD_DIR="$OUT_ROOT/shard_${TASK_ID}"
mkdir -p "$SHARD_DIR"

if (( START >= ${#PROMPTS[@]} )); then
  echo "No prompts assigned to task $TASK_ID" >&2
  exit 0
fi

for ((i=START; i<END && i<${#PROMPTS[@]}; i++)); do
  prompt_text="${PROMPTS[$i]}"
  prompt_dir="$SHARD_DIR/prompt_$(printf '%05d' "$i")"
  mkdir -p "$prompt_dir"
  python -m ascr.cli.run_stage1 \
    --config "$CONFIG" \
    --output-dir "$prompt_dir" \
    --prompt "$prompt_text"
done

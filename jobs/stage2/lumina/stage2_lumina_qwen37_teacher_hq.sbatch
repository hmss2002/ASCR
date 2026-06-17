#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage2-lumina-qwen37
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

CONFIG=${CONFIG:-configs/stage2/lumina/stage2_lumina_qwen37_teacher_hq.yaml}
OUT_ROOT=${OUT_ROOT:-outputs/stage2_lumina_qwen37_teacher_hq}
ASCR_ENV=${ASCR_ENV:-.venv-lumina}
PROMPT=${PROMPT:-}
PROMPT_FILE=${PROMPT_FILE:-}

if [[ -z "${OFOX_API_KEY:-}" ]]; then
  echo "OFOX_API_KEY is required for Stage 2 teacher runs." >&2
  exit 2
fi
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV" >&2
  exit 2
fi

source "$ASCR_ENV/bin/activate"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

run_one() {
  local prompt_text="$1"
  local shard_dir="$2"
  mkdir -p "$shard_dir"
  python -m ascr.cli.run_stage1 \
    --config "$CONFIG" \
    --output-dir "$shard_dir" \
    --prompt "$prompt_text"
}

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "PROMPT_FILE not found: $PROMPT_FILE" >&2
    exit 2
  fi
  index=0
  while IFS= read -r prompt_text || [[ -n "$prompt_text" ]]; do
    [[ -z "${prompt_text// }" ]] && continue
    run_one "$prompt_text" "$OUT_ROOT/file_run_$(printf '%04d' "$index")"
    index=$((index + 1))
  done < "$PROMPT_FILE"
  exit 0
fi

if [[ -z "$PROMPT" ]]; then
  echo "Set PROMPT or PROMPT_FILE before submitting this job." >&2
  exit 2
fi

run_one "$PROMPT" "$OUT_ROOT/single_prompt"

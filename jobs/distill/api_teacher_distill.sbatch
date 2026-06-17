#!/usr/bin/env bash
#SBATCH --job-name=ascr-api-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && ( -e "$SLURM_SUBMIT_DIR/pyproject.toml" || -e "$SLURM_SUBMIT_DIR/.git" ) ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
  fi
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT
mkdir -p logs

ASCR_ENV_QWEN=${ASCR_ENV_QWEN:-.venv-qwen36}
if [[ "$ASCR_ENV_QWEN" != /* ]]; then
  ASCR_ENV_QWEN="$PROJECT_ROOT/$ASCR_ENV_QWEN"
fi
if [[ ! -x "$ASCR_ENV_QWEN/bin/python" ]]; then
  echo "Missing ASCR_ENV_QWEN python at $ASCR_ENV_QWEN/bin/python" >&2
  exit 2
fi

export ASCR_TEACHER_MODEL=${ASCR_TEACHER_MODEL:-bailian/qwen3.7-plus}
export OFOX_BASE_URL=${OFOX_BASE_URL:-https://api.ofox.ai/v1}
export OUT_ROOT=${OUT_ROOT:-outputs/lumina_qwen_hard64}
export DISTILL_OUT=${DISTILL_OUT:-outputs/teacher_distill/hard64_lumina_qwen}
export LIMIT=${LIMIT:-64}
export WORKERS=${WORKERS:-1}

if [[ -z "${OFOX_API_KEY:-}" ]]; then
  echo "OFOX_API_KEY is not set. Submit with --export=ALL,OFOX_API_KEY,ASCR_TEACHER_MODEL=bailian/qwen3.7-plus,..." >&2
  exit 2
fi

PYTHONPATH="$PROJECT_ROOT" "$ASCR_ENV_QWEN/bin/python" scripts/distill/api_probe.py --model "$ASCR_TEACHER_MODEL" --base-url "$OFOX_BASE_URL"
PYTHONPATH="$PROJECT_ROOT" "$ASCR_ENV_QWEN/bin/python" -m ascr.distill.teacher \
  --out-root "$OUT_ROOT" \
  --output-dir "$DISTILL_OUT" \
  --limit "$LIMIT" \
  --workers "$WORKERS" \
  --model "$ASCR_TEACHER_MODEL" \
  --base-url "$OFOX_BASE_URL"

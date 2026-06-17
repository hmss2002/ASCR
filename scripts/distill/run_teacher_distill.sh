#!/usr/bin/env bash
# Run API teacher distillation labels from existing Stage-1 outputs.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

OUT_ROOT=${OUT_ROOT:-outputs/lumina_qwen_hard64}
DISTILL_OUT=${DISTILL_OUT:-outputs/teacher_distill/hard64_lumina_qwen}
LIMIT=${LIMIT:-64}
WORKERS=${WORKERS:-1}
ASCR_TEACHER_MODEL=${ASCR_TEACHER_MODEL:-bailian/qwen3.7-plus}
OFOX_BASE_URL=${OFOX_BASE_URL:-https://api.ofox.ai/v1}
ASCR_TEACHER_QUALITY_MAX_TOKENS=${ASCR_TEACHER_QUALITY_MAX_TOKENS:-2048}
ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=${ASCR_TEACHER_LOCALIZATION_MAX_TOKENS:-2048}
PATH_MODE=${PATH_MODE:-relative}

if [[ -z "${OFOX_API_KEY:-}" ]]; then
  echo "ERROR: OFOX_API_KEY is not set. Export it in the shell or pass it through Slurm; never write it into a tracked file." >&2
  exit 2
fi

export ASCR_TEACHER_MODEL OFOX_BASE_URL ASCR_TEACHER_QUALITY_MAX_TOKENS ASCR_TEACHER_LOCALIZATION_MAX_TOKENS

python -m ascr.distill.teacher \
  --out-root "$OUT_ROOT" \
  --output-dir "$DISTILL_OUT" \
  --limit "$LIMIT" \
  --workers "$WORKERS" \
  --model "$ASCR_TEACHER_MODEL" \
  --base-url "$OFOX_BASE_URL" \
  --quality-max-tokens "$ASCR_TEACHER_QUALITY_MAX_TOKENS" \
  --localization-max-tokens "$ASCR_TEACHER_LOCALIZATION_MAX_TOKENS" \
  --path-mode "$PATH_MODE"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-outputs/lumina_qwen_hard64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_lumina_native/sft_smoke}"
LIMIT="${LIMIT:-10}"

python -m ascr.training.train_lumina_evaluator \
  --dataset "$DATASET" \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --mode prepare-only \
  "$@"

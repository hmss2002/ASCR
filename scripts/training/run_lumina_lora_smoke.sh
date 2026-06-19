#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_JSONL="${DATA_JSONL:-outputs/stage2_lumina_native/lumina_sft_data_v3/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_lumina_native/lora_v3_clean_allmask}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-5}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
ANSWER_MASK_MODE="${ANSWER_MASK_MODE:-all}"

python -m ascr.training.train_lumina_lora_smoke \
  --repo-path "${LUMINA_REPO:-third_party/Lumina-DiMOO}" \
  --checkpoint-path "${LUMINA_MODEL_PATH:-models/lumina-dimoo}" \
  --data-jsonl "$DATA_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --image-size "$IMAGE_SIZE" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --answer-mask-mode "$ANSWER_MASK_MODE" \
  --ignore-pad-labels \
  --lora-r "${LORA_R:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --seed "${SEED:-0}" \
  "$@"

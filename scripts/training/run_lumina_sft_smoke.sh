#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LUMINA_REPO="${LUMINA_REPO:-third_party/Lumina-DiMOO}"
CHECKPOINT="${LUMINA_MODEL_PATH:-models/lumina-dimoo}"
DATA_CONFIG="${DATA_CONFIG:-configs/stage2/lumina/sft_smoke_data.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_lumina_native/sft_smoke/checkpoint}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$LUMINA_REPO:$PYTHONPATH"

python -u "$LUMINA_REPO/train/train.py" \
  --init_from "$CHECKPOINT" \
  --data_config "$DATA_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --dropout 0.05 \
  --wd 0.1 \
  --warmup_epochs 0.001 \
  --clip_grad 4 \
  --accum_iter 1 \
  --save_iteration_interval 100 \
  --num_workers 0 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-outputs/lumina_qwen_hard64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_lumina_native/json_probe}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"

if [[ -z "${IMAGE:-}" || -z "${PROMPT:-}" ]]; then
  mapfile -t probe_pair < <(python - "$DATASET" "$IMAGE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
image_root = Path(sys.argv[2])
row = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
loc = row["localizations"][0]
print(image_root / loc["grid_image"])
print(loc.get("prompt") or row.get("prompt") or "")
PY
)
  IMAGE="${IMAGE:-${probe_pair[0]}}"
  PROMPT="${PROMPT:-${probe_pair[1]}}"
fi

python -m ascr.cli.lumina_native_json_probe \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --repo-path "${LUMINA_REPO:-third_party/Lumina-DiMOO}" \
  --checkpoint-path "${LUMINA_MODEL_PATH:-models/lumina-dimoo}" \
  --device "${DEVICE:-cuda}" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --answer-steps "${ANSWER_STEPS:-64}" \
  --answer-block-length "${ANSWER_BLOCK_LENGTH:-128}" \
  --answer-temperature "${ANSWER_TEMPERATURE:-0.0}" \
  --answer-cfg-scale "${ANSWER_CFG_SCALE:-0.0}" \
  "$@"

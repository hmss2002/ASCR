#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-outputs/stage2_lumina_native/audit}"
mkdir -p "$OUT_DIR"

python -m ascr.cli.lumina_native_audit \
  --repo-path "${LUMINA_REPO:-third_party/Lumina-DiMOO}" \
  --checkpoint-path "${LUMINA_MODEL_PATH:-models/lumina-dimoo}" \
  --device "${DEVICE:-cuda}" \
  --scan-repo \
  --output "$OUT_DIR/audit.json" \
  --answer-steps "${ANSWER_STEPS:-64}" \
  --answer-block-length "${ANSWER_BLOCK_LENGTH:-128}" \
  --answer-temperature "${ANSWER_TEMPERATURE:-0.0}" \
  --answer-cfg-scale "${ANSWER_CFG_SCALE:-0.0}" \
  "$@"

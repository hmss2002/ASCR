#!/usr/bin/env bash
# Run or summarize a Stage-4 prompt/decoding sweep for a trained LoRA adapter.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid4_vq_tokens_l40s_1024px_gc.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_sweep_l40s_1024px_gc}
PROMPT_VARIANTS=${PROMPT_VARIANTS:-default,minimal_json,schema_first,schema_example}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128,384}
ANSWER_STEPS=${ANSWER_STEPS:-64}
ANSWER_TEMPERATURES=${ANSWER_TEMPERATURES:-0.0}
ANSWER_CFG_SCALES=${ANSWER_CFG_SCALES:-0.0}
ANSWER_BLOCK_LENGTHS=${ANSWER_BLOCK_LENGTHS:-128}
MODE=${MODE:-run}  # run, plan, summarize

args=(
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --prompt-variants "$PROMPT_VARIANTS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --answer-steps "$ANSWER_STEPS"
  --answer-temperatures "$ANSWER_TEMPERATURES"
  --answer-cfg-scales "$ANSWER_CFG_SCALES"
  --answer-block-lengths "$ANSWER_BLOCK_LENGTHS"
)

case "$MODE" in
  plan)
    "$PYTHON_BIN" -m ascr.cli.stage4_probe_sweep "${args[@]}" --write-plan-only
    ;;
  summarize)
    "$PYTHON_BIN" -m ascr.cli.stage4_probe_sweep "${args[@]}" --summarize-only
    ;;
  run)
    "$PYTHON_BIN" -m ascr.cli.stage4_probe_sweep "${args[@]}"
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# Run Stage-5 self-corruption loop smoke or benchmark.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi
MODE=${MODE:-loop}  # loop, benchmark, compare
CONFIG=${CONFIG:-configs/stage5/self_corrupt/token_repair_8x8.yaml}
PROMPT=${PROMPT:-"a green bench and a blue bowl"}
PROMPT_FILE=${PROMPT_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/stage5_self_corrupt/token_repair_8x8_smoke_epoch3}
LIMIT=${LIMIT:-16}
MOCK_FLAG=${MOCK_FLAG:-}

case "$MODE" in
  loop)
    "$PYTHON_BIN" -m ascr.cli.stage5_self_corrupt_loop \
      --prompt "$PROMPT" \
      --config "$CONFIG" \
      --output-dir "$OUTPUT_DIR" \
      $MOCK_FLAG
    ;;
  benchmark)
    "$PYTHON_BIN" -m ascr.cli.stage5_self_corrupt_benchmark \
      --prompts "$PROMPT_FILE" \
      --domain hard64_self_corrupt \
      --config "${BENCHMARK_CONFIG:-$CONFIG}" \
      --limit "$LIMIT" \
      --keep-going \
      --output-dir "$OUTPUT_DIR" \
      $MOCK_FLAG
    ;;
  compare)
    "$PYTHON_BIN" -m ascr.cli.stage5_compare_loop_results \
      --manifest "$OUTPUT_DIR/manifest.jsonl" \
      --output-dir "$OUTPUT_DIR/comparison"
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

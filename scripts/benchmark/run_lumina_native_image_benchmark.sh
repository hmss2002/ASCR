#!/usr/bin/env bash
# Run before/after image benchmarks with Lumina-native semantic evaluation.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PROMPTS=${PROMPTS:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
DOMAIN=${DOMAIN:-in_domain_hard64}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/image_bench/lumina_native/${DOMAIN}}
CONFIG=${CONFIG:-configs/stage2/lumina/lumina_native_evaluator_smoke.yaml}
GENERATOR=${GENERATOR:-lumina}
LIMIT=${LIMIT:-}
MAX_ITERATIONS=${MAX_ITERATIONS:-3}
SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
SHARD_COUNT=${SHARD_COUNT:-1}
PYTHON_BIN=${PYTHON_BIN:-}

if [[ -n "${OFOX_API_KEY:-}" ]]; then
  echo "ERROR: run_lumina_native_image_benchmark.sh must not receive OFOX_API_KEY; API judging belongs on the login node." >&2
  exit 2
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "ERROR: no Python executable found. Set PYTHON_BIN=/path/to/python." >&2
    exit 2
  fi
fi

args=(
  -m ascr.benchmarks.lumina_native_benchmark
  --prompts "$PROMPTS"
  --domain "$DOMAIN"
  --output-dir "$OUTPUT_DIR"
  --config "$CONFIG"
  --generator "$GENERATOR"
  --max-iterations "$MAX_ITERATIONS"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
  --keep-going
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" "${args[@]}"

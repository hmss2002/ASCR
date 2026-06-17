#!/usr/bin/env bash
# Run offline selector benchmarks without API or GPU dependencies.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

SELECTOR=${SELECTOR:-outputs/stage2_baselines/cell_prior_qwen37_holdout/selector_prior.json}
IN_DOMAIN_DATASET=${IN_DOMAIN_DATASET:-outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl}
IN_DOMAIN_SPLIT=${IN_DOMAIN_SPLIT:-outputs/stage2_baselines/cell_prior_qwen37_holdout/split_manifest.json}
OUT_DOMAIN_PROMPTS=${OUT_DOMAIN_PROMPTS:-configs/benchmarks/prompts/drawbench_smoke8.txt}
OUT_DOMAIN_LIMIT=${OUT_DOMAIN_LIMIT:-8}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/selector_benchmarks/cell_prior_qwen37}
TOP_K=${TOP_K:-3}
PYTHON_BIN=${PYTHON_BIN:-}

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v py >/dev/null 2>&1; then
    PYTHON_BIN="py -3"
  else
    echo "ERROR: no Python executable found. Set PYTHON_BIN=/path/to/python." >&2
    exit 2
  fi
fi

$PYTHON_BIN -m ascr.benchmarks.selector_benchmark \
  --selector "$SELECTOR" \
  --in-domain-dataset "$IN_DOMAIN_DATASET" \
  --in-domain-split "$IN_DOMAIN_SPLIT" \
  --out-domain-prompts "$OUT_DOMAIN_PROMPTS" \
  --out-domain-limit "$OUT_DOMAIN_LIMIT" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K"

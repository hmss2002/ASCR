#!/usr/bin/env bash
# Run before/after image benchmarks with the distilled student localizer.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

STUDENT_MODEL=${STUDENT_MODEL:-outputs/stage2_students/grid_localizer_v0/student_model.json}
PROMPTS=${PROMPTS:-outputs/stage2_students/grid_localizer_v0/holdout_prompts.txt}
DOMAIN=${DOMAIN:-in_domain_hard64_holdout}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/image_bench/student_localizer_v0/${DOMAIN}}
CONFIG=${CONFIG:-configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml}
GENERATOR=${GENERATOR:-lumina}
LIMIT=${LIMIT:-}
MAX_ITERATIONS=${MAX_ITERATIONS:-3}
SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
SHARD_COUNT=${SHARD_COUNT:-1}
PYTHON_BIN=${PYTHON_BIN:-}

if [[ -n "${OFOX_API_KEY:-}" ]]; then
  echo "ERROR: run_student_image_benchmark.sh must not receive OFOX_API_KEY; API judging belongs on the login node." >&2
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
  -m ascr.benchmarks.image_quality_benchmark
  --student-model "$STUDENT_MODEL"
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

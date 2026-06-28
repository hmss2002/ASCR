#!/usr/bin/env bash
# Submit and optionally merge Stage-3 locality probe shards.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

CONFIG=${CONFIG:-configs/stage3/self_corrupt/locality_probe_smoke.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/stage3_self_corrupt/locality_probe_parallel}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_COUNT=${PROMPT_COUNT:-}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-4}
WAIT=${WAIT:-0}
MERGE_AFTER=${MERGE_AFTER:-0}
BUILD_DATASET_AFTER=${BUILD_DATASET_AFTER:-0}
DATASET_OUTPUT_DIR=${DATASET_OUTPUT_DIR:-${OUTPUT_ROOT}_dataset}

PYTHON_BIN=${PYTHON_BIN:-python}
mkdir -p logs "$OUTPUT_ROOT"

if [[ -z "$PROMPT_COUNT" ]]; then
  if [[ -z "$PROMPT_FILE" ]]; then
    PROMPT_FILE=$("$PYTHON_BIN" - <<'PY' "$CONFIG"
import sys
from ascr.core.config import load_config
cfg = load_config(sys.argv[1])
print(cfg.get("prompt_file", ""))
PY
)
  fi
  if [[ -z "$PROMPT_FILE" ]]; then
    echo "PROMPT_COUNT or PROMPT_FILE is required" >&2
    exit 2
  fi
  PROMPT_COUNT=$("$PYTHON_BIN" - <<'PY' "$PROMPT_FILE"
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")))
PY
)
fi

if [[ "$PROMPT_COUNT" -le 0 ]]; then
  echo "PROMPT_COUNT must be > 0" >&2
  exit 2
fi

TASK_COUNT=$(( (PROMPT_COUNT + PROMPTS_PER_TASK - 1) / PROMPTS_PER_TASK ))
ARRAY_MAX=$(( TASK_COUNT - 1 ))

echo "Submitting ${TASK_COUNT} locality shards for ${PROMPT_COUNT} prompts (${PROMPTS_PER_TASK} prompts/task)."
JOB_ID=$(sbatch --parsable \
  --array=0-"$ARRAY_MAX" \
  --export=ALL,CONFIG="$CONFIG",OUTPUT_ROOT="$OUTPUT_ROOT",PROMPTS_PER_TASK="$PROMPTS_PER_TASK",PROMPT_FILE="$PROMPT_FILE" \
  jobs/stage3/self_corrupt_locality_probe_array.sbatch)
echo "$JOB_ID" | tee "$OUTPUT_ROOT/slurm_job_id.txt"

if [[ "$WAIT" == "1" ]]; then
  echo "Waiting for Slurm job $JOB_ID to leave the queue."
  while squeue -j "$JOB_ID" -h | grep -q .; do
    sleep 20
  done
fi

if [[ "$MERGE_AFTER" == "1" ]]; then
  "$PYTHON_BIN" -m ascr.cli.stage3_merge_probe_shards \
    --shard-dirs "$OUTPUT_ROOT"/shard_* \
    --output-dir "$OUTPUT_ROOT"
fi

if [[ "$BUILD_DATASET_AFTER" == "1" ]]; then
  "$PYTHON_BIN" -m ascr.cli.stage3_self_corrupt_dataset \
    --manifest "$OUTPUT_ROOT/manifest.jsonl" \
    --summary "$OUTPUT_ROOT/summary.json" \
    --output-dir "$DATASET_OUTPUT_DIR"
fi

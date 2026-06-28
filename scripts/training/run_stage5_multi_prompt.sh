#!/usr/bin/env bash
# Run independent Stage-5 self-corruption loops across GPUs in one allocation.

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

CONFIG=${CONFIG:-configs/stage5/self_corrupt/ascr_loop_smoke.yaml}
PROMPT_FILE=${PROMPT_FILE:-configs/benchmarks/prompts/t2i_compbench_hard64.txt}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/stage5_self_corrupt/multi_prompt}
GPU_COUNT=${GPU_COUNT:-$(bash scripts/slurm/dynamic_gpu_detect.sh --count)}
GPU_COUNT=${GPU_COUNT:-1}
PROMPTS_PER_GPU=${PROMPTS_PER_GPU:-1}
PROMPT_OFFSET=${PROMPT_OFFSET:-0}
MODE=${MODE:-run}  # run, summarize
MOCK_FLAG=${MOCK_FLAG:-}

mkdir -p "$OUTPUT_ROOT" logs

read_prompt() {
  "$PYTHON_BIN" - "$PROMPT_FILE" "$1" <<'PY'
import sys
from pathlib import Path

prompts = [
    line.strip()
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if not prompts:
    raise SystemExit(f"No prompts found in {sys.argv[1]}")
print(prompts[int(sys.argv[2]) % len(prompts)])
PY
}

if [[ "$MODE" == "summarize" ]]; then
  "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for trace_path in sorted(root.glob("prompt_*/trace.json")):
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    rows.append({
        "sample_index": int(trace_path.parent.name.split("_")[-1]),
        "prompt": trace.get("prompt"),
        "trace": str(trace_path),
        "clean_image": trace.get("clean_image"),
        "corrupted_image": trace.get("corrupted_image"),
        "repaired_image": trace.get("repaired_image"),
        "target_cells": trace.get("target_cells"),
        "lora_cells": trace.get("lora_cells"),
        "mask_stats": trace.get("mask_stats"),
        "reopen_changed": trace.get("reopen_changed"),
        "status": "ok",
    })
manifest = root / "manifest.jsonl"
with manifest.open("w", encoding="utf-8") as handle:
    for row in rows:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")
print(manifest)
PY
  "$PYTHON_BIN" -m ascr.cli.stage5_compare_loop_results \
    --manifest "$OUTPUT_ROOT/manifest.jsonl" \
    --output-dir "$OUTPUT_ROOT/comparison"
  exit 0
fi

pids=()
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for slot in $(seq 0 $((PROMPTS_PER_GPU - 1))); do
      index=$((PROMPT_OFFSET + gpu * PROMPTS_PER_GPU + slot))
      prompt=$(read_prompt "$index")
      sample_dir="$OUTPUT_ROOT/prompt_$(printf "%04d" "$index")"
      mkdir -p "$sample_dir"
      printf '%s\n' "$prompt" >"$sample_dir/prompt.txt"
      MODE=loop CONFIG="$CONFIG" PROMPT="$prompt" OUTPUT_DIR="$sample_dir" MOCK_FLAG="$MOCK_FLAG" \
        bash scripts/training/run_stage5_loop.sh
    done
  ) >"logs/stage5-mprompt-gpu-${gpu}.out" 2>"logs/stage5-mprompt-gpu-${gpu}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  MODE=summarize bash scripts/training/run_stage5_multi_prompt.sh
fi
exit "$status"

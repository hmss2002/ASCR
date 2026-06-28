#!/usr/bin/env bash
# Local/mock Stage 3 -> Stage 5 -> Stage 6 wiring smoke.

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
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/local_smoke/stage3_to_stage5_e2e}
PROMPTS="$OUTPUT_ROOT/prompts.txt"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" -m ascr.cli.stage3_sample_prompts \
  --sources configs/benchmarks/prompts/t2i_compbench_hard64.txt \
  --count 4 \
  --stratify complexity \
  --seed 0 \
  --output "$PROMPTS"

"$PYTHON_BIN" -m ascr.cli.stage5_self_corrupt_benchmark \
  --prompts "$PROMPTS" \
  --domain mock_self_corrupt \
  --config configs/stage5/self_corrupt/benchmark_hard64.yaml \
  --limit 2 \
  --keep-going \
  --mock \
  --output-dir "$OUTPUT_ROOT/stage5_benchmark"

"$PYTHON_BIN" -m ascr.cli.stage5_compare_loop_results \
  --manifest "$OUTPUT_ROOT/stage5_benchmark/manifest.jsonl" \
  --output-dir "$OUTPUT_ROOT/stage5_benchmark/comparison"

"$PYTHON_BIN" -m ascr.cli.stage6_transfer_probe \
  --prompts "$PROMPTS" \
  --limit 2 \
  --mock \
  --config configs/stage6/transfer_probe.yaml \
  --output-dir "$OUTPUT_ROOT/stage6_transfer"

"$PYTHON_BIN" -m ascr.analysis.stage6_transfer_metrics \
  --synthetic-manifest "$OUTPUT_ROOT/stage5_benchmark/manifest.jsonl" \
  --transfer-manifest "$OUTPUT_ROOT/stage6_transfer/manifest.jsonl" \
  --output-dir "$OUTPUT_ROOT/transfer_metrics"

"$PYTHON_BIN" -m ascr.cli.stage6_multi_arm_benchmark \
  --arms direct stage4_mmu_lora \
  --prompts "$PROMPTS" \
  --limit 2 \
  --mock \
  --output-dir "$OUTPUT_ROOT/multi_arm_plan"

echo "E2E mock outputs: $OUTPUT_ROOT"

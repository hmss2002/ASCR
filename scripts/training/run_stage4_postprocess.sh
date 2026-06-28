#!/usr/bin/env bash
# Build Stage-4 registries, optional failure summaries, and next-action hints.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
ROOTS=${ROOTS:-outputs/stage4_self_corrupt}
REGISTRY_DIR=${REGISTRY_DIR:-outputs/stage4_self_corrupt/registry}
NEXT_DIR=${NEXT_DIR:-outputs/stage4_self_corrupt/next_actions}
LOG_GLOB=${LOG_GLOB:-logs/ascr-s4-*.out logs/ascr-s4-*.err}

"$PYTHON_BIN" -m ascr.cli.stage4_build_run_registry \
  --roots $ROOTS \
  --output-dir "$REGISTRY_DIR"

maybe_analyze_grid4_gc() {
  local probe_dir=outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval
  local probe_rows="$probe_dir/probe_rows.jsonl"
  if [[ -s "$probe_rows" ]]; then
    "$PYTHON_BIN" -m ascr.cli.stage4_analyze_probe_failures \
      --probe-rows "$probe_rows" \
      --summary "$probe_dir/summary.json" \
      --sft-examples outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/sft/train_sft_examples.jsonl \
      --train-jsonl outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/lumina_sft/train.jsonl \
      --output-dir "$probe_dir/failure_analysis"
  else
    echo "No grid4 GC probe rows found at $probe_rows; skipping failure analysis."
  fi
}

maybe_analyze_grid4_gc

decision_args=(
  --registry "$REGISTRY_DIR/stage4_run_registry.json"
  --failure-summary "outputs/stage4_self_corrupt/**/failure_analysis/failure_summary.json"
  --output-dir "$NEXT_DIR"
)

for pattern in $LOG_GLOB; do
  decision_args+=(--log "$pattern")
done

"$PYTHON_BIN" -m ascr.cli.stage4_decide_next "${decision_args[@]}"

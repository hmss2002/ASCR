#!/usr/bin/env bash
# Single-process ASCR Stage-1 entrypoint. Use --dry-run for a no-model local check.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)}
cd "$PROJECT_ROOT"

CONFIG=${CONFIG:-configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml}
PROMPT=${PROMPT:-"A red cube left of a blue sphere"}
OUT_ROOT=${OUT_ROOT:-outputs/single_inference}
MAX_ITERS=${MAX_ITERS:-1}
GENERATOR=${GENERATOR:-}
EVALUATOR=${EVALUATOR:-}
DRY_RUN=${DRY_RUN:-0}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

cmd=(python -m ascr.cli.run_stage1 --config "$CONFIG" --prompt "$PROMPT" --output-dir "$OUT_ROOT" --max-iterations "$MAX_ITERS")
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ -n "$GENERATOR" ]]; then
  cmd+=(--generator "$GENERATOR")
fi
if [[ -n "$EVALUATOR" ]]; then
  cmd+=(--evaluator "$EVALUATOR")
fi

printf '+'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"

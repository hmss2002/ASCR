#!/usr/bin/env bash
# Generate or execute the Stage-4 schema_example 1024px-GC server campaign.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/stage4_self_corrupt/campaigns/stage4_1024gc_schema_example}
PROFILE=${PROFILE:-l40s_1024_gc}
GRIDS=${GRIDS:-4,8,16}
MODE=${MODE:-plan}  # plan, submit_curriculum, split_curriculum, summarize, diagnostic_sweep

"$PYTHON_BIN" -m ascr.cli.stage4_server_campaign \
  --output-dir "$OUTPUT_DIR" \
  --profile "$PROFILE" \
  --grids "$GRIDS"

case "$MODE" in
  plan)
    cat "$OUTPUT_DIR/campaign_plan.md"
    ;;
  submit_curriculum|split_curriculum|summarize|diagnostic_sweep)
    MODE="$MODE" bash "$OUTPUT_DIR/run_stage4_server_campaign.sh"
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

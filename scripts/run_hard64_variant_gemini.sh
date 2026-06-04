#!/usr/bin/env bash
set -euo pipefail

# Login-node orchestration of Gemini judging for the Hard64 variant comparison.
# Compute nodes have NO internet; run this ONLY on a login node (e.g. hpcr4300a).
#
# Steps:
#   1. collect each arm's comparison.json into a manifest.json
#   2. clean pass/fail for baseline + direct + coarse
#   3. bidirectional pairwise direct vs coarse
#   4. print summary
#
# Required env:
#   OFOX_API_KEY        ofox.ai key (env ONLY; never written to a file)
#   DIRECT_RUN_ROOT     run root produced by the direct-arm generation jobs
#   COARSE_RUN_ROOT     run root produced by the coarse-arm generation jobs
# Optional:
#   OUT_DIR             where to write manifests + judgements (default outputs/hard64_variant_judge_<stamp>)
#   WORKERS             concurrent API workers (default 8)
#   LIMIT               cap prompts (smoke test)

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$PROJECT_ROOT"

if [[ -z "${OFOX_API_KEY:-}" ]]; then
  echo "ERROR: OFOX_API_KEY is not set. Export it in your shell (never commit it)." >&2
  echo "  export OFOX_API_KEY=sk-..." >&2
  exit 2
fi

ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ -d "$ASCR_ENV" ]]; then
  source "$ASCR_ENV/bin/activate"
fi
PYTHON_BIN=${PYTHON_BIN:-python}

DIRECT_RUN_ROOT=${DIRECT_RUN_ROOT:?Set DIRECT_RUN_ROOT to the direct-arm generation output root}
COARSE_RUN_ROOT=${COARSE_RUN_ROOT:?Set COARSE_RUN_ROOT to the coarse-arm generation output root}
WORKERS=${WORKERS:-8}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-outputs/hard64_variant_judge_${STAMP}}
LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

mkdir -p "$OUT_DIR"

echo "[judge] collecting manifests"
"$PYTHON_BIN" scripts/collect_variant_images.py --run-root "$DIRECT_RUN_ROOT" --arm direct_token \
  --output "$OUT_DIR/manifest_direct.json"
"$PYTHON_BIN" scripts/collect_variant_images.py --run-root "$COARSE_RUN_ROOT" --arm coarse_grid \
  --output "$OUT_DIR/manifest_coarse.json"

echo "[judge] clean: baseline (from direct manifest baseline_image)"
"$PYTHON_BIN" scripts/judge_variants_gemini.py clean \
  --items-file "$OUT_DIR/manifest_direct.json" --image-field baseline_image \
  --arm-label baseline --output "$OUT_DIR/clean_baseline.json" --workers "$WORKERS" "${LIMIT_ARG[@]}"

echo "[judge] clean: direct"
"$PYTHON_BIN" scripts/judge_variants_gemini.py clean \
  --items-file "$OUT_DIR/manifest_direct.json" --image-field final_image \
  --arm-label direct --output "$OUT_DIR/clean_direct.json" --workers "$WORKERS" "${LIMIT_ARG[@]}"

echo "[judge] clean: coarse"
"$PYTHON_BIN" scripts/judge_variants_gemini.py clean \
  --items-file "$OUT_DIR/manifest_coarse.json" --image-field final_image \
  --arm-label coarse --output "$OUT_DIR/clean_coarse.json" --workers "$WORKERS" "${LIMIT_ARG[@]}"

echo "[judge] pairwise: direct vs coarse (bidirectional)"
"$PYTHON_BIN" scripts/judge_variants_gemini.py pairwise \
  --direct-manifest "$OUT_DIR/manifest_direct.json" \
  --coarse-manifest "$OUT_DIR/manifest_coarse.json" \
  --output "$OUT_DIR/pairwise_direct_vs_coarse.json" --workers "$WORKERS" "${LIMIT_ARG[@]}"

echo ""
echo "============ Hard64 Variant Comparison Summary ============"
"$PYTHON_BIN" - "$OUT_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def load(name):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else {}
for label, fn in [("baseline", "clean_baseline.json"), ("direct", "clean_direct.json"), ("coarse", "clean_coarse.json")]:
    d = load(fn)
    print(f"clean {label:8s}: pass_rate={d.get('pass_rate')}%  counts={d.get('counts')}")
pw = load("pairwise_direct_vs_coarse.json")
print(f"pairwise: direct_wins={pw.get('direct_wins')} coarse_wins={pw.get('coarse_wins')} ties={pw.get('ties')} "
      f"direct_decisive_winrate={pw.get('direct_win_rate_decisive')}%")
print(f"output dir: {out}")
PYEOF

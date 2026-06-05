#!/usr/bin/env bash
# Run after stage1_geneval_generate_8gpu.sbatch (job 68794) finishes successfully.
# Discovers the new RUN_ROOT and submits 3 scoring jobs (ShowO50/ASCR50/BAGEL).
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

GEN_JOB=${GEN_JOB:?GEN_JOB env var required (numeric jobid of geneval-gen-8gpu)}

# Find the run root created by the gen job (latest matching directory).
RUN_ROOT=$(ls -dt outputs/geneval_showo_ascr_${GEN_JOB}_*/ 2>/dev/null | head -1 | sed 's:/$::')
if [[ -z "$RUN_ROOT" || ! -d "$RUN_ROOT/geneval_baseline" || ! -d "$RUN_ROOT/geneval_ascr" ]]; then
  echo "ERROR: cannot locate GenEval run root for job $GEN_JOB (looked for outputs/geneval_showo_ascr_${GEN_JOB}_*)" >&2
  exit 1
fi
echo "[submit_score] RUN_ROOT=$RUN_ROOT"

# Locate the BAGEL GenEval baseline images (created by stage1_geneval_bagel_generate.sbatch).
BAGEL_RUN=$(ls -dt outputs/geneval_bagel_*/geneval_bagel 2>/dev/null | head -1 || true)
if [[ -z "$BAGEL_RUN" || ! -d "$BAGEL_RUN" ]]; then
  echo "WARNING: BAGEL GenEval images not found; skipping BAGEL scoring." >&2
  BAGEL_RUN=""
fi
echo "[submit_score] BAGEL=$BAGEL_RUN"

JOB1=$(GENEVAL_IMAGES="$RUN_ROOT/geneval_baseline" GENEVAL_LABEL=ShowO50 sbatch --parsable jobs/benchmarks/stage1_geneval_score_single.sbatch)
echo "  ShowO50 score: $JOB1"
JOB2=$(GENEVAL_IMAGES="$RUN_ROOT/geneval_ascr"     GENEVAL_LABEL=ASCR50  sbatch --parsable jobs/benchmarks/stage1_geneval_score_single.sbatch)
echo "  ASCR50  score: $JOB2"
if [[ -n "$BAGEL_RUN" ]]; then
  JOB3=$(GENEVAL_IMAGES="$BAGEL_RUN" GENEVAL_LABEL=BAGEL sbatch --parsable jobs/benchmarks/stage1_geneval_score_single.sbatch)
  echo "  BAGEL   score: $JOB3"
fi
echo "[submit_score] Done."

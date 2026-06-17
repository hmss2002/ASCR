#!/usr/bin/env bash
# Submit the supported single-node multi-GPU Stage-1 jobs through Slurm.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)}
cd "$PROJECT_ROOT"

MODE=${MODE:-lumina-qwen}
PROMPT_LIMIT=${PROMPT_LIMIT:-1}
OUT_ROOT=${OUT_ROOT:-outputs/multigpu_smoke}

case "$MODE" in
  lumina-qwen)
    JOB=${JOB:-jobs/smoke/stage1_lumina_qwen_2gpu.sbatch}
    ;;
  lumina-qwen-8gpu)
    JOB=${JOB:-jobs/stage1/lumina/stage1_lumina_qwen_coarse_hard64_8gpu.sbatch}
    PROMPT_LIMIT=${PROMPT_LIMIT:-64}
    OUT_ROOT=${OUT_ROOT:-outputs/lumina_qwen_hard64}
    ;;
  mmada-self)
    JOB=${JOB:-jobs/smoke/stage1_mmada_self_1gpu.sbatch}
    ;;
  *)
    echo "Unknown MODE=$MODE. Use lumina-qwen, lumina-qwen-8gpu, or mmada-self." >&2
    exit 2
    ;;
esac

export PROMPT_LIMIT OUT_ROOT
mkdir -p logs

echo "+ PROMPT_LIMIT=$PROMPT_LIMIT OUT_ROOT=$OUT_ROOT sbatch $JOB"
exec sbatch "$JOB"

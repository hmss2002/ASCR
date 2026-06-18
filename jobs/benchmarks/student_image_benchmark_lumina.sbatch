#!/usr/bin/env bash
#SBATCH --job-name=ascr-student-img
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -n "${OFOX_API_KEY:-}" ]]; then
  echo "ERROR: compute-node image benchmark must not receive OFOX_API_KEY." >&2
  exit 2
fi

if [[ -d .venv-lumina ]]; then
  source .venv-lumina/bin/activate
elif [[ -d .venv-qwen36 ]]; then
  source .venv-qwen36/bin/activate
fi

export PYTHON_BIN=${PYTHON_BIN:-python}
export STUDENT_MODEL=${STUDENT_MODEL:-outputs/stage2_students/grid_localizer_v0/student_model.json}
export PROMPTS=${PROMPTS:-configs/benchmarks/prompts/geneval_553.txt}
export DOMAIN=${DOMAIN:-geneval_smoke16}
export LIMIT=${LIMIT:-16}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/image_bench/student_localizer_v0/${DOMAIN}}
export CONFIG=${CONFIG:-configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml}
export GENERATOR=${GENERATOR:-lumina}
export MAX_ITERATIONS=${MAX_ITERATIONS:-3}
export SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
export SHARD_COUNT=${SHARD_COUNT:-1}

bash scripts/benchmark/run_student_image_benchmark.sh

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

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && ( -e "$SLURM_SUBMIT_DIR/pyproject.toml" || -e "$SLURM_SUBMIT_DIR/.git" ) ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
  fi
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT
mkdir -p logs

leaked_api_env=()
for name in OFOX_API_KEY OFOX_BASE_URL ASCR_TEACHER_MODEL ASCR_TEACHER_QUALITY_MAX_TOKENS ASCR_TEACHER_LOCALIZATION_MAX_TOKENS ASCR_TEACHER_JSON_REPAIR_RETRIES; do
  if [[ -n "${!name:-}" ]]; then
    leaked_api_env+=("$name")
  fi
done
if (( ${#leaked_api_env[@]} )); then
  echo "WARNING: clearing OFOX/API judge environment variables from compute-node image benchmark job: ${leaked_api_env[*]}" >&2
fi
unset OFOX_API_KEY OFOX_BASE_URL ASCR_TEACHER_MODEL ASCR_TEACHER_QUALITY_MAX_TOKENS ASCR_TEACHER_LOCALIZATION_MAX_TOKENS ASCR_TEACHER_JSON_REPAIR_RETRIES

if [[ -d .venv-lumina ]]; then
  source .venv-lumina/bin/activate
elif [[ -d .venv-qwen36 ]]; then
  source .venv-qwen36/bin/activate
fi

export PYTHON_BIN=${PYTHON_BIN:-python}
export STUDENT_MODEL=${STUDENT_MODEL:-outputs/stage2_students/grid_localizer_v0/student_model.json}
export PROMPTS=${PROMPTS:-configs/benchmarks/prompts/geneval_553.txt}
export DOMAIN=${DOMAIN:-geneval_smoke16}
export LIMIT=${LIMIT:-}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/image_bench/student_localizer_v0/${DOMAIN}}
export CONFIG=${CONFIG:-configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml}
export GENERATOR=${GENERATOR:-lumina}
export MAX_ITERATIONS=${MAX_ITERATIONS:-3}
export SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
export SHARD_COUNT=${SHARD_COUNT:-1}

bash scripts/benchmark/run_student_image_benchmark.sh

#!/usr/bin/env bash
#SBATCH --job-name=ascr-lumina-native-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if [[ -d ".venv-lumina" ]]; then
  source .venv-lumina/bin/activate
elif [[ -d ".venv-qwen36" ]]; then
  source .venv-qwen36/bin/activate
fi

if [[ -n "${OFOX_API_KEY:-}" ]]; then
  echo "ERROR: OFOX_API_KEY must not be exported to compute-node benchmark jobs." >&2
  exit 2
fi

export SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
export SHARD_COUNT="${SHARD_COUNT:-1}"

bash scripts/benchmark/run_lumina_native_image_benchmark.sh "$@"

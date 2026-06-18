#!/usr/bin/env bash
#SBATCH --job-name=ascr-lumina-native-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if [[ -d ".venv-lumina" ]]; then
  source .venv-lumina/bin/activate
elif [[ -d ".venv-qwen36" ]]; then
  source .venv-qwen36/bin/activate
fi

bash scripts/training/run_lumina_native_audit.sh "$@"

#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage3-locality
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

python -m ascr.cli.token_locality_probe \
  --config "${CONFIG:-configs/stage3/self_corrupt/locality_probe_smoke.yaml}" \
  "$@"

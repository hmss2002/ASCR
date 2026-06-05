#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage2-selector
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs checkpoints

ASCR_ENV=${ASCR_ENV:-.venv-qwen36}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV." >&2; exit 2
fi
source "$ASCR_ENV/bin/activate"

torchrun --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} -m ascr.training.train_selector --config configs/stage1/showo/stage1_showo_local.yaml

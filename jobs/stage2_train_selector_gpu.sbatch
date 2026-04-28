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
source .venv/bin/activate
torchrun --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} -m ascr.training.train_selector --config configs/stage1_showo_local.yaml

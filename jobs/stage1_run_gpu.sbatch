#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/slurm_gpu
source .venv/bin/activate
python -m ascr.cli.run_stage1 --generator showo --evaluator showo_mmu --config configs/stage1_showo_local.yaml --output-dir outputs/slurm_gpu

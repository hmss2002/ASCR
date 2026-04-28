#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage1-debug
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/slurm_gpu_shared
source .venv/bin/activate
python -m ascr.cli.run_stage1 --dry-run --generator mock --evaluator mock --config configs/stage1_showo_local.yaml --output-dir outputs/slurm_gpu_shared

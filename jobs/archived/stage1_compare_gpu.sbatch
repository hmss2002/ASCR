#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage1-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/benchmarks
source .venv/bin/activate
bash scripts/run_stage1_showo_compare.sh

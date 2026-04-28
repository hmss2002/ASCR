#!/usr/bin/env bash
#SBATCH --job-name=ascr-stage1-compare-shared
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs outputs/benchmarks_gpu_shared
source .venv/bin/activate
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/benchmarks_gpu_shared}
export GENERATION_TIMESTEPS=${GENERATION_TIMESTEPS:-4}
export MAX_ITERATIONS=${MAX_ITERATIONS:-2}
bash scripts/run_stage1_showo_compare.sh

#!/usr/bin/env bash
# Generic Slurm wrapper for ASCR Stage-1 inference smoke/full runs.
#
# Examples:
#   MODE=lumina-qwen PROMPT_LIMIT=1 sbatch scripts/slurm_infer.sbatch
#   MODE=mmada-self PROMPT_LIMIT=1 sbatch scripts/slurm_infer.sbatch
#
# For the 8-GPU mode, request matching resources at submission time:
#   MODE=lumina-qwen-8gpu PROMPT_LIMIT=64 \
#     sbatch --gres=gpu:8 --cpus-per-task=64 --mem=400G --time=08:00:00 scripts/slurm_infer.sbatch

#SBATCH --job-name=ascr-infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && ( -e "$SLURM_SUBMIT_DIR/pyproject.toml" || -e "$SLURM_SUBMIT_DIR/.git" ) ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
  fi
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT

MODE=${MODE:-lumina-qwen}
case "$MODE" in
  lumina-qwen)
    exec bash jobs/smoke/stage1_lumina_qwen_2gpu.sbatch
    ;;
  lumina-qwen-8gpu)
    exec bash jobs/stage1/lumina/stage1_lumina_qwen_coarse_hard64_8gpu.sbatch
    ;;
  mmada-self)
    exec bash jobs/smoke/stage1_mmada_self_1gpu.sbatch
    ;;
  *)
    echo "Unknown MODE=$MODE. Use lumina-qwen, lumina-qwen-8gpu, or mmada-self." >&2
    exit 2
    ;;
esac

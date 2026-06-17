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
DATASET_PATH=${DATASET_PATH:-outputs/stage2_lumina_qwen37_teacher_hq/stage2_teacher_dataset.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/stage2_selector_replay}
TRAIN_MODE=${TRAIN_MODE:-dataset_replay}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}
CONFIG=${CONFIG:-}
LIMIT_ARG=()
CONFIG_ARG=()
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing ASCR environment at $ASCR_ENV." >&2; exit 2
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Missing Stage 2 dataset at $DATASET_PATH." >&2; exit 2
fi
source "$ASCR_ENV/bin/activate"
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi
if [[ -n "$CONFIG" ]]; then
  CONFIG_ARG=(--config "$CONFIG")
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m ascr.training.train_selector \
  "${CONFIG_ARG[@]}" \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$TRAIN_MODE" \
  "${LIMIT_ARG[@]}"

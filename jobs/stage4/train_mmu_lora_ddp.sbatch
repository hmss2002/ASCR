#!/usr/bin/env bash
#SBATCH --job-name=ascr-s4-lora-ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && ( -e "$SLURM_SUBMIT_DIR/pyproject.toml" || -e "$SLURM_SUBMIT_DIR/.git" ) ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
  fi
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT
mkdir -p logs

ASCR_ENV=${ASCR_ENV:-.venv-lumina}
if [[ "$ASCR_ENV" != /* ]]; then
  ASCR_ENV="$PROJECT_ROOT/$ASCR_ENV"
fi
source "$ASCR_ENV/bin/activate"

export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ASCR_DDP_DEBUG=${ASCR_DDP_DEBUG:-1}
export ASCR_DDP_IGNORE_FROZEN=${ASCR_DDP_IGNORE_FROZEN:-1}
export ASCR_DDP_IGNORE_FROZEN_METHOD=${ASCR_DDP_IGNORE_FROZEN_METHOD:-attribute}
export ASCR_DDP_INIT_SYNC=${ASCR_DDP_INIT_SYNC:-0}
export ASCR_DDP_BROADCAST_BUFFERS=${ASCR_DDP_BROADCAST_BUFFERS:-0}
export ASCR_DDP_FIND_UNUSED_PARAMETERS=${ASCR_DDP_FIND_UNUSED_PARAMETERS:-0}
export ASCR_DDP_GRADIENT_AS_BUCKET_VIEW=${ASCR_DDP_GRADIENT_AS_BUCKET_VIEW:-1}
CONFIG=${CONFIG:-configs/stage4/self_corrupt/mmu_lora_train_hard64_grid8_vq_tokens_l40s_1024px_gc_adam8bit.yaml}
NPROC=${NPROC:-$(bash scripts/slurm/dynamic_gpu_detect.sh --count)}
TORCHRUN_ARGS=(--config "$CONFIG")
[[ -n "${DATA_JSONL:-}" ]] && TORCHRUN_ARGS+=(--data-jsonl "$DATA_JSONL")
[[ -n "${VAL_JSONL:-}" ]] && TORCHRUN_ARGS+=(--val-jsonl "$VAL_JSONL")
[[ -n "${OUTPUT_DIR:-}" ]] && TORCHRUN_ARGS+=(--output-dir "$OUTPUT_DIR")
[[ -n "${EPOCHS:-}" ]] && TORCHRUN_ARGS+=(--epochs "$EPOCHS")
[[ -n "${LIMIT:-}" ]] && TORCHRUN_ARGS+=(--limit "$LIMIT")
[[ -n "${RESUME_FROM_ADAPTER:-}" ]] && TORCHRUN_ARGS+=(--resume-from-adapter "$RESUME_FROM_ADAPTER")
[[ -n "${CHECKPOINT_EVERY_EPOCHS:-}" ]] && TORCHRUN_ARGS+=(--checkpoint-every-epochs "$CHECKPOINT_EVERY_EPOCHS")
[[ -n "${EARLY_STOPPING_PATIENCE:-}" ]] && TORCHRUN_ARGS+=(--early-stopping-patience "$EARLY_STOPPING_PATIENCE")
[[ -n "${EARLY_STOPPING_MIN_DELTA:-}" ]] && TORCHRUN_ARGS+=(--early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA")

# Real DDP: each rank owns one full Lumina+LoRA replica, uses DistributedSampler,
# and rank 0 is the only writer.
torchrun --nproc_per_node="$NPROC" -m ascr.cli.stage4_train_mmu_lora_ddp "${TORCHRUN_ARGS[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=ascr-mmada-self-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-GPU smoke test for the MMaDA-8B self-selection Stage-1 task.
# Verifies: 8B loads once (local_files_only), t2i_generate produces an image,
# mmu_generate self-evaluation returns parseable cells, force_mask+reopen works.

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

export ASCR_ENV=${ASCR_ENV:-.venv-mmada}
if [[ ! -x "$ASCR_ENV/bin/python" ]]; then
  echo "Missing MMaDA environment at $ASCR_ENV." >&2
  exit 2
fi
export PYTHON_BIN="$ASCR_ENV/bin/python"
unset PYTHONPATH
export PYTHONPATH="."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON_BIN" -c "import torch, transformers; print('transformers', transformers.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

PROMPT=${PROMPT:-"A red cube on the left and a blue sphere on the right"}
OUT=${OUT:-outputs/stage1_mmada_self_smoke}

"$PYTHON_BIN" -m ascr.cli.run_stage1_mmada_self \
  --config configs/stage1/mmada/stage1_mmada8b_self_direct_token.yaml \
  --prompt "$PROMPT" \
  --output-dir "$OUT" \
  --max-iterations 2

echo "SMOKE DONE -> $OUT"
ls -la "$OUT" || true

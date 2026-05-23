#!/usr/bin/env bash
#SBATCH --job-name=bench3-eval
#SBATCH --partition=gpu_shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Run the full bench3 GPT-5.5 evaluation pipeline (DPG, DSG-1k, GenAI-Bench).
# CPU-only: no GPU needed; gpu_shared accepts jobs without --gres.
#
# Required env vars:
#   SHOWO_RUN_ROOT  - bench3 ShowO+ASCR generation run root (contains merged suite.json after 68941)
#   BAGEL_RUN_ROOT  - bench3 BAGEL generation run root (contains shard_*/images/)
#   OFOX_API_KEY    - ofox.ai API key for GPT-5.5

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

SHOWO_RUN_ROOT="${SHOWO_RUN_ROOT:?Must set SHOWO_RUN_ROOT}"
BAGEL_RUN_ROOT="${BAGEL_RUN_ROOT:?Must set BAGEL_RUN_ROOT}"

if [[ -z "${OFOX_API_KEY:-}" ]]; then
  echo "ERROR: OFOX_API_KEY is not set" >&2; exit 2
fi

VENV="${VENV:-.venv-qwen36}"
source "$VENV/bin/activate"
PYTHON="$VENV/bin/python"

SHOWO_SUITE="$SHOWO_RUN_ROOT/suite.json"
if [[ ! -f "$SHOWO_SUITE" ]]; then
  echo "ERROR: $SHOWO_SUITE not found — run after 68941 (geneval-merge-eval) completes" >&2; exit 2
fi

EVAL_ROOT="${EVAL_ROOT:-outputs/bench3_eval}"
mkdir -p "$EVAL_ROOT"

echo "[bench3-eval] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[bench3-eval] SHOWO_RUN_ROOT=$SHOWO_RUN_ROOT"
echo "[bench3-eval] BAGEL_RUN_ROOT=$BAGEL_RUN_ROOT"
echo "[bench3-eval] EVAL_ROOT=$EVAL_ROOT"

# ── Step 1: build image map ───────────────────────────────────────────────────
IMAGE_MAP="$EVAL_ROOT/image_map.json"
echo "[bench3-eval] Step 1: building image map → $IMAGE_MAP"
"$PYTHON" scripts/build_bench_image_map.py \
  --showo-suite "$SHOWO_SUITE" \
  --bagel-run-root "$BAGEL_RUN_ROOT" \
  --bench3-index configs/benchmark_data/bench3_index.json \
  --output "$IMAGE_MAP"

# ── Step 2: DPG-Bench (3 models in parallel) ─────────────────────────────────
echo "[bench3-eval] Step 2: DPG-Bench evaluation (3 models in parallel)"
for model in showo ascr bagel; do
  "$PYTHON" scripts/eval_csv_vqa_gpt.py \
    --csv configs/benchmark_data/dpg_bench.csv \
    --image-map "$IMAGE_MAP" \
    --model-key "$model" \
    --output-dir "$EVAL_ROOT/dpg_${model}" \
    --workers 8 &
done
wait
echo "[bench3-eval] DPG-Bench done."

# ── Step 3: DSG-1k (3 models in parallel) ────────────────────────────────────
echo "[bench3-eval] Step 3: DSG-1k evaluation (3 models in parallel)"
for model in showo ascr bagel; do
  "$PYTHON" scripts/eval_csv_vqa_gpt.py \
    --csv configs/benchmark_data/dsg1k_anns.csv \
    --image-map "$IMAGE_MAP" \
    --model-key "$model" \
    --output-dir "$EVAL_ROOT/dsg_${model}" \
    --workers 8 &
done
wait
echo "[bench3-eval] DSG-1k done."

# ── Step 4: GenAI-Bench (3 models in parallel) ───────────────────────────────
echo "[bench3-eval] Step 4: GenAI-Bench evaluation (3 models in parallel)"
for model in showo ascr bagel; do
  "$PYTHON" scripts/eval_genai_gpt.py \
    --image-map "$IMAGE_MAP" \
    --model-key "$model" \
    --output-dir "$EVAL_ROOT/genai_${model}" \
    --workers 8 &
done
wait
echo "[bench3-eval] GenAI-Bench done."

# ── Step 5: summarize ─────────────────────────────────────────────────────────
echo "[bench3-eval] Step 5: summarizing → outputs/bench3_summary.json"
"$PYTHON" scripts/summarize_bench3.py \
  --eval-root "$EVAL_ROOT" \
  --output outputs/bench3_summary.json

echo "[bench3-eval] All done. Results: outputs/bench3_summary.json"

#!/usr/bin/env bash
# Re-run all 9 bench3 eval tasks using google/gemini-3-flash-preview.
# Runs on the login node (requires outbound internet).
# Usage: bash scripts/rerun_bench3_gemini.sh
# Must be run from /grp01/cds_bdai/JianyuZhang/ASCR with venv active.

set -e
export OFOX_API_KEY="sk-of-bTORRveHyXdWZyGqRbgddEJDJuAPVPpnIlTPNMKCxJiygQDvvtWlhjJBZglXzihp"
MODEL="google/gemini-3-flash-preview"
WORKERS=30
IMAGE_MAP="outputs/bench3_eval/image_map.json"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== Launching 9 eval processes with $MODEL ==="

# --- DPG-Bench ---
for mk in showo ascr bagel; do
  nohup python scripts/eval_csv_vqa_gpt.py \
    --csv configs/benchmark_data/dpg_bench.csv \
    --image-map "$IMAGE_MAP" \
    --model-key "$mk" \
    --output-dir "outputs/bench3_eval/dpg_${mk}" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    > "$LOG_DIR/eval_dpg_${mk}_gemini.log" 2>&1 &
  echo "  dpg_${mk} PID=$!"
done

# --- DSG-1k ---
for mk in showo ascr bagel; do
  nohup python scripts/eval_csv_vqa_gpt.py \
    --csv configs/benchmark_data/dsg1k_anns.csv \
    --image-map "$IMAGE_MAP" \
    --model-key "$mk" \
    --output-dir "outputs/bench3_eval/dsg_${mk}" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    > "$LOG_DIR/eval_dsg_${mk}_gemini.log" 2>&1 &
  echo "  dsg_${mk} PID=$!"
done

# --- GenAI-Bench ---
for mk in showo ascr bagel; do
  nohup python scripts/eval_genai_gpt.py \
    --metadata configs/benchmark_data/genai_bench.jsonl \
    --image-map "$IMAGE_MAP" \
    --model-key "$mk" \
    --output-dir "outputs/bench3_eval/genai_${mk}" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    > "$LOG_DIR/eval_genai_${mk}_gemini.log" 2>&1 &
  echo "  genai_${mk} PID=$!"
done

echo ""
echo "All 9 processes launched. Monitor with:"
echo "  tail -f logs/eval_dpg_showo_gemini.log"
echo "  watch -n30 'tail -1 logs/eval_*_gemini.log'"

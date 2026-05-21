#!/usr/bin/env bash
#SBATCH --job-name=hard64-bagel-3way-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /grp01/cds_bdai/JianyuZhang/ASCR
mkdir -p logs

source "${ASCR_ENV:-.venv-qwen36}/bin/activate"
PY="${ASCR_ENV:-.venv-qwen36}/bin/python"
export PYTHONPATH=".deps/transformers-qwen35-clean/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

: "${HARD64_RUN:?HARD64_RUN must point to the new hard64 ASCR run root}"
: "${BAGEL_DIR:?BAGEL_DIR must point to the BAGEL hard64 run root}"
CONFIG="${CONFIG:-configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml}"

HARD64_SUITE="$HARD64_RUN/suite.json"
BAGEL_SUITE="$BAGEL_DIR/bagel_vs_ascr_suite.json"

OUT_ROOT="$HARD64_RUN/bagel_3way"
mkdir -p "$OUT_ROOT"

echo "[3way] hard64 suite: $HARD64_SUITE"
echo "[3way] bagel suite : $BAGEL_SUITE"

for SIDE in baseline ascr; do
  PAIRED="$OUT_ROOT/bagel_vs_${SIDE}_suite.json"
  JUDGE="$OUT_ROOT/qwen_pairwise_bagel_vs_${SIDE}.json"
  LEFT_LABEL="$([[ $SIDE == baseline ]] && echo 'ShowO' || echo 'ASCR')"

  echo "[3way] pair LEFT=$LEFT_LABEL RIGHT=BAGEL"
  "$PY" scripts/pair_bagel_vs_hard64_run.py \
      --bagel-suite "$BAGEL_SUITE" \
      --hard64-suite "$HARD64_SUITE" \
      --side "$SIDE" \
      --output "$PAIRED"

  echo "[3way] judge LEFT=$LEFT_LABEL RIGHT=BAGEL → $JUDGE"
  "$PY" scripts/judge_showo_ascr_pairwise_qwen.py \
      "$PAIRED" \
      --config "$CONFIG" \
      --output "$JUDGE" \
      --baseline-label "$LEFT_LABEL" \
      --ascr-label "BAGEL" \
      --no-image-labels
done

echo "[3way] Done. Outputs under $OUT_ROOT/"

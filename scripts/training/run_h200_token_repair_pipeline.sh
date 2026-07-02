#!/usr/bin/env bash
# Submit the current-server H200 fast path for Stage-3 token repair through
# Stage-4 LoRA training/probing.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"
mkdir -p logs

ASCR_ENV=${ASCR_ENV:-.venv-lumina}
PROFILE=${PROFILE:-h200_1024}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-313}
CLEAN_OUTPUT_ROOT=${CLEAN_OUTPUT_ROOT:-outputs/stage3_token_repair/clean_tokens_h200_32x4}
CLEAN_MANIFEST=${CLEAN_MANIFEST:-$CLEAN_OUTPUT_ROOT/clean_manifest.jsonl}
CLEAN_JOB_ID=${CLEAN_JOB_ID:-}
MODE=${MODE:-submit}

if [[ "$ASCR_ENV" != /* ]]; then
  ASCR_ENV_PATH="$PROJECT_ROOT/$ASCR_ENV"
else
  ASCR_ENV_PATH="$ASCR_ENV"
fi

parse_job_id() {
  awk '/Submitted batch job/ {print $4}' | tail -n 1
}

case "$MODE" in
  submit)
    if [[ -z "$CLEAN_JOB_ID" ]]; then
      clean_submit=$(
        MODE=submit_clean_h200 \
          CLEAN_OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT" \
          PROMPTS_PER_TASK="$PROMPTS_PER_TASK" \
          bash scripts/training/run_stage3_token_repair_dataset.sh
      )
      echo "$clean_submit"
      CLEAN_JOB_ID=$(printf "%s\n" "$clean_submit" | parse_job_id)
    fi
    if [[ -z "$CLEAN_JOB_ID" ]]; then
      echo "Could not determine Stage-3 clean-token job id." >&2
      exit 3
    fi

    sbatch --dependency=afterok:"$CLEAN_JOB_ID" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ascr-h200-build-s4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=$PROJECT_ROOT/logs/%x-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x-%j.err

set -euo pipefail
cd "$PROJECT_ROOT"
source "$ASCR_ENV_PATH/bin/activate"
export PROJECT_ROOT="$PROJECT_ROOT"
export PYTHON_BIN="$ASCR_ENV_PATH/bin/python"
export LUMINA_REPO=third_party/Lumina-DiMOO
export LUMINA_MODEL_PATH=models/lumina-dimoo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CLEAN_OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT"
export CLEAN_MANIFEST="$CLEAN_MANIFEST"

MODE=merge_clean bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=report_clean REPORT_MIN_ROWS=10000 bash scripts/training/run_stage3_token_repair_dataset.sh
clean_rows=\$(wc -l < "\$CLEAN_MANIFEST")
echo "[stage3] clean manifest rows: \$clean_rows"
if [[ "\$clean_rows" -lt 10000 ]]; then
  echo "[stage3] expected at least 10000 clean rows, got \$clean_rows" >&2
  exit 4
fi

MODE=build_dataset bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=prepare_sft PROFILE="$PROFILE" bash scripts/training/run_stage4_token_repair_lora.sh
MODE=convert_sft PROFILE="$PROFILE" DEVICE=cpu bash scripts/training/run_stage4_token_repair_lora.sh

train_submit=\$(MODE=submit_train PROFILE="$PROFILE" ASCR_ENV="$ASCR_ENV_PATH" bash scripts/training/run_stage4_token_repair_lora.sh)
echo "\$train_submit"
train_job=\$(awk '/Submitted batch job/ {print \$4}' <<<"\$train_submit" | tail -n 1)
if [[ -z "\$train_job" ]]; then
  echo "[stage4] could not parse train job id" >&2
  exit 5
fi
echo "[stage4] train job: \$train_job"

sbatch --dependency=afterok:"\$train_job" \
  --job-name=ascr-h200-probe-lora \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=80G \
  --time=03:00:00 \
  --output="$PROJECT_ROOT/logs/%x-%j.out" \
  --error="$PROJECT_ROOT/logs/%x-%j.err" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",ASCR_ENV="$ASCR_ENV_PATH",PROFILE="$PROFILE",LUMINA_REPO=third_party/Lumina-DiMOO,LUMINA_MODEL_PATH=models/lumina-dimoo,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,TOKENIZERS_PARALLELISM=false \
  --wrap='cd "\$PROJECT_ROOT" && source "\$ASCR_ENV/bin/activate" && MODE=speed_report PROFILE="\$PROFILE" bash scripts/training/run_stage4_token_repair_lora.sh && MODE=probe_lora PROFILE="\$PROFILE" bash scripts/training/run_stage4_token_repair_lora.sh'
EOF
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

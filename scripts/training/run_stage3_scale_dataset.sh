#!/usr/bin/env bash
# Scale Stage-3 self-corruption prompt sampling and locality dataset construction.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi
MODE=${MODE:-plan}  # plan, sample, submit, merge, build_dataset, all_local
PROMPT_COUNT=${PROMPT_COUNT:-256}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-4}
CONFIG=${CONFIG:-configs/stage3/self_corrupt/locality_probe_hard256.yaml}
PROMPT_OUTPUT=${PROMPT_OUTPUT:-configs/benchmarks/prompts/stage3_hard256_sampled.txt}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/stage3_self_corrupt/locality_probe_hard256}
DATASET_OUTPUT_DIR=${DATASET_OUTPUT_DIR:-outputs/stage3_self_corrupt/datasets/locality_hard256_v1}
SOURCES=${SOURCES:-"configs/benchmarks/prompts/dpg_bench_1065.txt configs/benchmarks/prompts/genai_bench_1600.txt configs/benchmarks/prompts/dsg1k_1060.txt"}
HOLDOUT=${HOLDOUT:-configs/benchmarks/prompts/geneval_553.txt}

print_plan() {
  cat <<CMDS
# 1. Sample prompts
MODE=sample PROMPT_COUNT=$PROMPT_COUNT bash scripts/training/run_stage3_scale_dataset.sh

# 2. Submit locality probe shards
MODE=submit CONFIG=$CONFIG OUTPUT_ROOT=$OUTPUT_ROOT PROMPT_COUNT=$PROMPT_COUNT PROMPTS_PER_TASK=$PROMPTS_PER_TASK bash scripts/training/run_stage3_scale_dataset.sh

# 3. Merge shards after Slurm completion
MODE=merge OUTPUT_ROOT=$OUTPUT_ROOT bash scripts/training/run_stage3_scale_dataset.sh

# 4. Build dataset
MODE=build_dataset OUTPUT_ROOT=$OUTPUT_ROOT DATASET_OUTPUT_DIR=$DATASET_OUTPUT_DIR bash scripts/training/run_stage3_scale_dataset.sh
CMDS
}

case "$MODE" in
  plan)
    print_plan
    ;;
  sample)
    "$PYTHON_BIN" -m ascr.cli.stage3_sample_prompts \
      --sources $SOURCES \
      --count "$PROMPT_COUNT" \
      --stratify complexity \
      --holdout "$HOLDOUT" \
      --seed "${SEED:-0}" \
      --output "$PROMPT_OUTPUT"
    ;;
  submit)
    PROMPT_FILE="$PROMPT_OUTPUT" PROMPT_COUNT="$PROMPT_COUNT" PROMPTS_PER_TASK="$PROMPTS_PER_TASK" \
      CONFIG="$CONFIG" OUTPUT_ROOT="$OUTPUT_ROOT" bash scripts/training/run_stage3_locality_parallel.sh
    ;;
  merge)
    "$PYTHON_BIN" -m ascr.cli.stage3_merge_probe_shards \
      --shard-dirs "$OUTPUT_ROOT"/shard_* \
      --output-dir "$OUTPUT_ROOT"
    ;;
  build_dataset)
    "$PYTHON_BIN" -m ascr.cli.stage3_self_corrupt_dataset \
      --manifest "$OUTPUT_ROOT/manifest.jsonl" \
      --summary "$OUTPUT_ROOT/summary.json" \
      --output-dir "$DATASET_OUTPUT_DIR"
    ;;
  all_local)
    MODE=sample bash scripts/training/run_stage3_scale_dataset.sh
    MODE=submit bash scripts/training/run_stage3_scale_dataset.sh
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

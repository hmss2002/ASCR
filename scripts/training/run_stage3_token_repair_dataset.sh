#!/usr/bin/env bash
# Build the canonical Stage-3 token-only 8x8 repair_cells dataset.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
MODE=${MODE:-plan}

RAW_PROMPTS=${RAW_PROMPTS:-outputs/stage3_token_repair/prompts/diffusiondb_prompts.jsonl}
PROMPT_OUTPUT=${PROMPT_OUTPUT:-configs/benchmarks/prompts/stage3_token_repair_prompts_10k.txt}
PROMPT_COUNT=${PROMPT_COUNT:-10000}
DIFFUSIONDB_LIMIT=${DIFFUSIONDB_LIMIT:-20000}
DIFFUSIONDB_SUBSET=${DIFFUSIONDB_SUBSET:-2m_first_10k}
SOURCES=${SOURCES:-"$RAW_PROMPTS configs/benchmarks/prompts/bench3_combined.txt configs/benchmarks/prompts/dpg_bench_1065.txt configs/benchmarks/prompts/dsg1k_1060.txt configs/benchmarks/prompts/genai_bench_1600.txt"}
HOLDOUT=${HOLDOUT:-configs/benchmarks/prompts/geneval_553.txt}

CLEAN_OUTPUT_ROOT=${CLEAN_OUTPUT_ROOT:-outputs/stage3_token_repair/clean_tokens}
CLEAN_MANIFEST=${CLEAN_MANIFEST:-$CLEAN_OUTPUT_ROOT/clean_manifest.jsonl}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-1024}
MULTINODE_NODES=${MULTINODE_NODES:-2}

DATASET_OUTPUT_DIR=${DATASET_OUTPUT_DIR:-outputs/stage3_token_repair/datasets/repair_cells_40k}
POSITIVE_ROWS=${POSITIVE_ROWS:-30000}
NEGATIVE_ROWS=${NEGATIVE_ROWS:-10000}
VARIANTS_PER_CLEAN=${VARIANTS_PER_CLEAN:-3}
MASK_SIZES=${MASK_SIZES:-"1 2 4 8"}
OPERATORS=${OPERATORS:-"random_replace local_shuffle neighbor_copy transplant"}
AUDIT_OUTPUT_DIR=${AUDIT_OUTPUT_DIR:-outputs/stage3_token_repair/audit_decode}
AUDIT_PAIRS=${AUDIT_PAIRS:-16}
SEED=${SEED:-0}

print_plan() {
  cat <<CMDS
# Prompt preparation
MODE=download_prompts bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=sample_prompts bash scripts/training/run_stage3_token_repair_dataset.sh

# Best path if the cluster allows one multi-node job, each node with 8 GPUs
MODE=submit_clean_multinode MULTINODE_NODES=$MULTINODE_NODES bash scripts/training/run_stage3_token_repair_dataset.sh

# Fallback: job array, one node per task, each node with 8 GPUs
MODE=submit_clean PROMPTS_PER_TASK=$PROMPTS_PER_TASK bash scripts/training/run_stage3_token_repair_dataset.sh

# Merge clean token manifests and build token-only repair dataset
MODE=merge_clean bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=build_dataset bash scripts/training/run_stage3_token_repair_dataset.sh

# Optional visual audit only; decoded images are not used for labels
MODE=audit_decode AUDIT_PAIRS=$AUDIT_PAIRS bash scripts/training/run_stage3_token_repair_dataset.sh
CMDS
}

case "$MODE" in
  plan)
    print_plan
    ;;
  download_prompts)
    mkdir -p "$(dirname "$RAW_PROMPTS")"
    "$PYTHON_BIN" -m ascr.cli.stage3_download_diffusiondb_prompts \
      --output "$RAW_PROMPTS" \
      --subset "$DIFFUSIONDB_SUBSET" \
      --limit "$DIFFUSIONDB_LIMIT"
    ;;
  sample_prompts)
    "$PYTHON_BIN" -m ascr.cli.stage3_sample_prompts \
      --sources $SOURCES \
      --count "$PROMPT_COUNT" \
      --stratify hard_first \
      --holdout "$HOLDOUT" \
      --seed "$SEED" \
      --output "$PROMPT_OUTPUT"
    ;;
  submit_clean)
    task_count=$(( (PROMPT_COUNT + PROMPTS_PER_TASK - 1) / PROMPTS_PER_TASK ))
    last_task=$(( task_count - 1 ))
    sbatch --array=0-"$last_task" \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",PROMPTS="$PROMPT_OUTPUT",OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT",PROMPTS_PER_TASK="$PROMPTS_PER_TASK",SEED="$SEED" \
      jobs/stage3/token_repair_clean_tokens.sbatch
    ;;
  submit_clean_multinode)
    sbatch --nodes="$MULTINODE_NODES" \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",PROMPTS="$PROMPT_OUTPUT",OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT",PROMPTS_PER_NODE="$PROMPTS_PER_TASK",SEED="$SEED" \
      jobs/stage3/token_repair_clean_tokens_multinode.sbatch
    ;;
  generate_clean_local)
    PROMPTS="$PROMPT_OUTPUT" OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT" PROMPT_LIMIT="$PROMPT_COUNT" \
      SEED="$SEED" bash scripts/training/run_stage3_clean_tokens_node.sh
    ;;
  merge_clean)
    mkdir -p "$(dirname "$CLEAN_MANIFEST")"
    find "$CLEAN_OUTPUT_ROOT" -path "*/manifest.jsonl" -type f | sort | xargs cat > "$CLEAN_MANIFEST"
    ;;
  build_dataset)
    "$PYTHON_BIN" -m ascr.cli.stage3_build_token_repair_dataset \
      --clean-manifest "$CLEAN_MANIFEST" \
      --output-dir "$DATASET_OUTPUT_DIR" \
      --positive-rows "$POSITIVE_ROWS" \
      --negative-rows "$NEGATIVE_ROWS" \
      --variants-per-clean "$VARIANTS_PER_CLEAN" \
      --mask-sizes $MASK_SIZES \
      --operators $OPERATORS \
      --action-grid-size 8 \
      --seed "$SEED" \
      --project-root "$PROJECT_ROOT"
    ;;
  audit_decode)
    "$PYTHON_BIN" -m ascr.cli.stage3_decode_token_repair_audit \
      --dataset "$DATASET_OUTPUT_DIR/dataset.jsonl" \
      --output-dir "$AUDIT_OUTPUT_DIR" \
      --max-pairs "$AUDIT_PAIRS" \
      --seed "$SEED" \
      --repo-path "${LUMINA_REPO:-third_party/Lumina-DiMOO}" \
      --checkpoint-path "${LUMINA_MODEL_PATH:-models/lumina-dimoo}"
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

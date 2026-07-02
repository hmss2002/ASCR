#!/usr/bin/env bash
# Build the canonical Stage-3 token-only 8x8 repair_cells dataset.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)}
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv-lumina/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv-lumina/bin/python"
  else
    PYTHON_BIN=python
  fi
fi
MODE=${MODE:-plan}

RAW_PROMPTS=${RAW_PROMPTS:-configs/benchmarks/prompts/diffusiondb_10k.jsonl}
RAW_PROMPT_TXT=${RAW_PROMPT_TXT:-configs/benchmarks/prompts/diffusiondb_10k.txt}
PROMPT_OUTPUT=${PROMPT_OUTPUT:-configs/benchmarks/prompts/stage3_token_repair_prompts_10k.txt}
PROMPT_COUNT=${PROMPT_COUNT:-10000}
DIFFUSIONDB_LIMIT=${DIFFUSIONDB_LIMIT:-10000}
DIFFUSIONDB_SUBSET=${DIFFUSIONDB_SUBSET:-2m_text_only}
SOURCES=${SOURCES:-"$RAW_PROMPTS configs/benchmarks/prompts/bench3_combined.txt configs/benchmarks/prompts/dpg_bench_1065.txt configs/benchmarks/prompts/dsg1k_1060.txt configs/benchmarks/prompts/genai_bench_1600.txt configs/benchmarks/prompts/drawbench_all.txt"}
HOLDOUT=${HOLDOUT:-configs/benchmarks/prompts/geneval_553.txt}

_CLEAN_OUTPUT_ROOT_WAS_SET=${CLEAN_OUTPUT_ROOT+x}
_PROMPTS_PER_TASK_WAS_SET=${PROMPTS_PER_TASK+x}
_CLEAN_GPUS_PER_TASK_WAS_SET=${CLEAN_GPUS_PER_TASK+x}
_CLEAN_CPUS_PER_TASK_WAS_SET=${CLEAN_CPUS_PER_TASK+x}
_CLEAN_MEM_WAS_SET=${CLEAN_MEM+x}
_CLEAN_TIME_WAS_SET=${CLEAN_TIME+x}

CLEAN_OUTPUT_ROOT=${CLEAN_OUTPUT_ROOT:-outputs/stage3_token_repair/clean_tokens}
CLEAN_MANIFEST=${CLEAN_MANIFEST:-$CLEAN_OUTPUT_ROOT/clean_manifest.jsonl}
PROMPTS_PER_TASK=${PROMPTS_PER_TASK:-1024}
MULTINODE_NODES=${MULTINODE_NODES:-2}
CLEAN_GPUS_PER_TASK=${CLEAN_GPUS_PER_TASK:-8}
CLEAN_CPUS_PER_TASK=${CLEAN_CPUS_PER_TASK:-32}
CLEAN_MEM=${CLEAN_MEM:-240G}
CLEAN_TIME=${CLEAN_TIME:-08:00:00}
CLEAN_ARRAY_LIMIT=${CLEAN_ARRAY_LIMIT:-}
CLEAN_SHARD_NODE_START=${CLEAN_SHARD_NODE_START:-0}
CLEAN_SHARD_NODE_END=${CLEAN_SHARD_NODE_END:-}
CLEAN_SHARDS_PER_NODE=${CLEAN_SHARDS_PER_NODE:-4}
CLEAN_SHARD_GPUS_PER_TASK=${CLEAN_SHARD_GPUS_PER_TASK:-1}
CLEAN_SHARD_CPUS_PER_TASK=${CLEAN_SHARD_CPUS_PER_TASK:-8}
CLEAN_SHARD_MEM=${CLEAN_SHARD_MEM:-80G}
CLEAN_SHARD_TIME=${CLEAN_SHARD_TIME:-08:00:00}
CLEAN_SHARD_ARRAY_LIMIT=${CLEAN_SHARD_ARRAY_LIMIT:-}

DATASET_OUTPUT_DIR=${DATASET_OUTPUT_DIR:-outputs/stage3_token_repair/datasets/repair_cells_40k}
POSITIVE_ROWS=${POSITIVE_ROWS:-30000}
NEGATIVE_ROWS=${NEGATIVE_ROWS:-10000}
VARIANTS_PER_CLEAN=${VARIANTS_PER_CLEAN:-3}
MASK_SIZES=${MASK_SIZES:-"1 2 4 8"}
OPERATORS=${OPERATORS:-"random_replace local_shuffle neighbor_copy transplant"}
AUDIT_OUTPUT_DIR=${AUDIT_OUTPUT_DIR:-outputs/stage3_token_repair/audit_decode}
AUDIT_PAIRS=${AUDIT_PAIRS:-16}
REPORT_OUTPUT_DIR=${REPORT_OUTPUT_DIR:-$CLEAN_OUTPUT_ROOT/report}
REPORT_MIN_ROWS=${REPORT_MIN_ROWS:-0}
SEED=${SEED:-0}
FORCE_DOWNLOAD=${FORCE_DOWNLOAD:-0}
FORCE_RESAMPLE=${FORCE_RESAMPLE:-0}

count_nonempty_lines() {
  if [[ -f "$1" ]]; then
    awk 'NF {count++} END {print count+0}' "$1"
  else
    echo 0
  fi
}

print_plan() {
  cat <<CMDS
# Prompt preparation is preseeded in git. These commands only refresh if forced
# or if the prompt files are missing.
MODE=download_prompts bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=sample_prompts bash scripts/training/run_stage3_token_repair_dataset.sh

# Best path if the cluster allows one multi-node job, each node with 8 GPUs
MODE=submit_clean_multinode MULTINODE_NODES=$MULTINODE_NODES bash scripts/training/run_stage3_token_repair_dataset.sh

# Fallback: job array, one node per task, each node with 8 GPUs
MODE=submit_clean PROMPTS_PER_TASK=$PROMPTS_PER_TASK bash scripts/training/run_stage3_token_repair_dataset.sh

# Current H200 server fast path: many short 4-GPU tasks keep the 16-GPU account
# allowance full and reduce idle tail time as tasks finish.
MODE=submit_clean_h200 bash scripts/training/run_stage3_token_repair_dataset.sh

# Opportunistic H200 tail path: split selected clean nodes into 1-GPU shards so
# leftover account quota can be used instead of waiting for a full 4-GPU slot.
MODE=submit_clean_h200_shards CLEAN_SHARD_NODE_START=24 CLEAN_SHARD_NODE_END=31 bash scripts/training/run_stage3_token_repair_dataset.sh

# Merge clean token manifests and build token-only repair dataset
MODE=merge_clean bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=report_clean REPORT_MIN_ROWS=10000 bash scripts/training/run_stage3_token_repair_dataset.sh
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
    existing=$(count_nonempty_lines "$RAW_PROMPTS")
    if [[ "$FORCE_DOWNLOAD" != "1" && "$existing" -ge "$DIFFUSIONDB_LIMIT" ]]; then
      echo "Using existing $RAW_PROMPTS ($existing rows); set FORCE_DOWNLOAD=1 to refresh."
      exit 0
    fi
    mkdir -p "$(dirname "$RAW_PROMPTS")"
    "$PYTHON_BIN" -m ascr.cli.stage3_download_diffusiondb_prompts \
      --output "$RAW_PROMPTS" \
      --subset "$DIFFUSIONDB_SUBSET" \
      --limit "$DIFFUSIONDB_LIMIT"
    ;;
  sample_prompts)
    existing=$(count_nonempty_lines "$PROMPT_OUTPUT")
    if [[ "$FORCE_RESAMPLE" != "1" && "$existing" -ge "$PROMPT_COUNT" ]]; then
      echo "Using existing $PROMPT_OUTPUT ($existing prompts); set FORCE_RESAMPLE=1 to rebuild."
      exit 0
    fi
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
    array_spec="0-$last_task"
    if [[ -n "$CLEAN_ARRAY_LIMIT" ]]; then
      array_spec="${array_spec}%${CLEAN_ARRAY_LIMIT}"
    fi
    sbatch --array="$array_spec" \
      --gres=gpu:"$CLEAN_GPUS_PER_TASK" \
      --cpus-per-task="$CLEAN_CPUS_PER_TASK" \
      --mem="$CLEAN_MEM" \
      --time="$CLEAN_TIME" \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",PROMPTS="$PROMPT_OUTPUT",OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT",PROMPTS_PER_TASK="$PROMPTS_PER_TASK",SEED="$SEED" \
      jobs/stage3/token_repair_clean_tokens.sbatch
    ;;
  submit_clean_h200)
    if [[ -z "${_CLEAN_OUTPUT_ROOT_WAS_SET:-}" ]]; then
      CLEAN_OUTPUT_ROOT=outputs/stage3_token_repair/clean_tokens_h200_32x4
    fi
    if [[ -z "${_CLEAN_GPUS_PER_TASK_WAS_SET:-}" ]]; then
      CLEAN_GPUS_PER_TASK=4
    fi
    if [[ -z "${_CLEAN_CPUS_PER_TASK_WAS_SET:-}" ]]; then
      CLEAN_CPUS_PER_TASK=16
    fi
    if [[ -z "${_CLEAN_MEM_WAS_SET:-}" ]]; then
      CLEAN_MEM=160G
    fi
    if [[ -z "${_CLEAN_TIME_WAS_SET:-}" ]]; then
      CLEAN_TIME=08:00:00
    fi
    if [[ -z "${_PROMPTS_PER_TASK_WAS_SET:-}" ]]; then
      PROMPTS_PER_TASK=313
    fi
    MODE=submit_clean \
      CLEAN_OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT" \
      CLEAN_GPUS_PER_TASK="$CLEAN_GPUS_PER_TASK" \
      CLEAN_CPUS_PER_TASK="$CLEAN_CPUS_PER_TASK" \
      CLEAN_MEM="$CLEAN_MEM" \
      CLEAN_TIME="$CLEAN_TIME" \
      PROMPTS_PER_TASK="$PROMPTS_PER_TASK" \
      bash scripts/training/run_stage3_token_repair_dataset.sh
    ;;
  submit_clean_h200_shards)
    if [[ -z "${_CLEAN_OUTPUT_ROOT_WAS_SET:-}" ]]; then
      CLEAN_OUTPUT_ROOT=outputs/stage3_token_repair/clean_tokens_h200_32x4
    fi
    if [[ -z "${_PROMPTS_PER_TASK_WAS_SET:-}" ]]; then
      PROMPTS_PER_TASK=313
    fi
    task_count=$(( (PROMPT_COUNT + PROMPTS_PER_TASK - 1) / PROMPTS_PER_TASK ))
    last_node=$(( task_count - 1 ))
    if [[ -z "$CLEAN_SHARD_NODE_END" ]]; then
      CLEAN_SHARD_NODE_END="$last_node"
    fi
    first_shard=$(( CLEAN_SHARD_NODE_START * CLEAN_SHARDS_PER_NODE ))
    last_shard=$(( (CLEAN_SHARD_NODE_END + 1) * CLEAN_SHARDS_PER_NODE - 1 ))
    array_spec="$first_shard-$last_shard"
    if [[ -n "$CLEAN_SHARD_ARRAY_LIMIT" ]]; then
      array_spec="${array_spec}%${CLEAN_SHARD_ARRAY_LIMIT}"
    fi
    sbatch --array="$array_spec" \
      --job-name=ascr-s3-token-shard \
      --partition=gpu \
      --gres=gpu:"$CLEAN_SHARD_GPUS_PER_TASK" \
      --cpus-per-task="$CLEAN_SHARD_CPUS_PER_TASK" \
      --mem="$CLEAN_SHARD_MEM" \
      --time="$CLEAN_SHARD_TIME" \
      --output="$PROJECT_ROOT/logs/%x-%A_%a.out" \
      --error="$PROJECT_ROOT/logs/%x-%A_%a.err" \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",PYTHON_BIN="$PYTHON_BIN",PROMPTS="$PROMPT_OUTPUT",CLEAN_OUTPUT_ROOT="$CLEAN_OUTPUT_ROOT",PROMPTS_PER_TASK="$PROMPTS_PER_TASK",PROMPT_COUNT="$PROMPT_COUNT",CLEAN_SHARDS_PER_NODE="$CLEAN_SHARDS_PER_NODE",SEED="$SEED" \
      --wrap='set -euo pipefail
cd "$PROJECT_ROOT"
shard_id=${SLURM_ARRAY_TASK_ID:?}
node_index=$(( shard_id / CLEAN_SHARDS_PER_NODE ))
shard_rank=$(( shard_id % CLEAN_SHARDS_PER_NODE ))
prompts_per_shard=$(( (PROMPTS_PER_TASK + CLEAN_SHARDS_PER_NODE - 1) / CLEAN_SHARDS_PER_NODE ))
node_start=$(( node_index * PROMPTS_PER_TASK ))
offset=$(( node_start + shard_rank * prompts_per_shard ))
node_stop=$(( node_start + PROMPTS_PER_TASK ))
if [[ "$node_stop" -gt "$PROMPT_COUNT" ]]; then
  node_stop="$PROMPT_COUNT"
fi
if [[ "$offset" -ge "$node_stop" ]]; then
  echo "[stage3-shard] shard $shard_id has no prompts; node=$node_index rank=$shard_rank offset=$offset stop=$node_stop"
  exit 0
fi
limit="$prompts_per_shard"
remaining=$(( node_stop - offset ))
if [[ "$remaining" -lt "$limit" ]]; then
  limit="$remaining"
fi
shard_dir="$CLEAN_OUTPUT_ROOT/node_$(printf "%04d" "$node_index")/gpu_$(printf "%02d" "$shard_rank")"
echo "[stage3-shard] shard=$shard_id node=$node_index rank=$shard_rank offset=$offset limit=$limit output=$shard_dir"
"$PYTHON_BIN" -m ascr.cli.stage3_generate_clean_tokens \
  --prompts "$PROMPTS" \
  --output-dir "$shard_dir" \
  --prompt-offset "$offset" \
  --prompt-limit "$limit" \
  --seed "$SEED" \
  --repo-path "${LUMINA_REPO:-third_party/Lumina-DiMOO}" \
  --checkpoint-path "${LUMINA_MODEL_PATH:-models/lumina-dimoo}" \
  --device cuda \
  --image-size "${IMAGE_SIZE:-1024}" \
  --token-grid-size "${TOKEN_GRID_SIZE:-64}" \
  --generation-timesteps "${GENERATION_TIMESTEPS:-64}" \
  --guidance-scale "${GUIDANCE_SCALE:-4.0}" \
  --temperature "${GENERATION_TEMPERATURE:-1.0}"'
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
    find "$CLEAN_OUTPUT_ROOT" -path "*/manifest.jsonl" -type f | sort | xargs -r cat > "$CLEAN_MANIFEST"
    ;;
  report_clean)
    "$PYTHON_BIN" -m ascr.cli.stage3_clean_manifest_report \
      --manifests "$CLEAN_MANIFEST" \
      --output-root "$CLEAN_OUTPUT_ROOT" \
      --project-root "$PROJECT_ROOT" \
      --min-rows "$REPORT_MIN_ROWS" \
      --output-dir "$REPORT_OUTPUT_DIR" \
      --strict
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

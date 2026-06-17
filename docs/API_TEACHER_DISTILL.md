# API Teacher Distillation

This guide describes how to use an OFOX/OpenAI-compatible API model as an ASCR
teacher. The teacher reads existing Stage-1 outputs and writes reusable
distillation labels. It does not train the Stage-2 selector yet.

## Secret Handling

Never write real API keys into tracked files. Set them only in the current shell
or scheduler environment:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048
export ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=2048
export ASCR_TEACHER_JSON_REPAIR_RETRIES=1
```

PowerShell:

```powershell
$env:OFOX_API_KEY="<your-ofox-api-key>"
$env:OFOX_BASE_URL="https://api.ofox.ai/v1"
$env:ASCR_TEACHER_MODEL="bailian/qwen3.7-plus"
$env:ASCR_TEACHER_QUALITY_MAX_TOKENS="2048"
$env:ASCR_TEACHER_LOCALIZATION_MAX_TOKENS="2048"
$env:ASCR_TEACHER_JSON_REPAIR_RETRIES="1"
```

## Inputs

The teacher expects a completed Stage-1 output root:

```text
outputs/lumina_qwen_hard64/
  records/p000.json
  baseline/p000.png
  self/p000.png
  runs/p000/<stage1-run>/trace.jsonl
```

Each `trace.jsonl` provides grid-image paths for localization labels. Each
record provides baseline/final image paths for quality labels.

## Outputs

Default output directory:

```text
outputs/teacher_distill/hard64_lumina_qwen/
  localization_labels.jsonl
  quality_labels.jsonl
  manifest.json
  errors.jsonl
```

`localization_labels.jsonl` stores teacher semantic-grid labels compatible with
`SemanticEvaluation`. `quality_labels.jsonl` stores baseline-vs-final teacher
scores and winner labels.

## Local API Probe

```bash
python scripts/distill/api_probe.py --allow-empty-content
```

This sends one tiny request and prints the model/base URL plus a short response
preview. It never prints the API key. For `bailian/qwen3.7-plus`, a tiny
text-only probe can return empty content even when the main teacher run works;
Slurm uses `--allow-empty-content` so that this false negative is only a
warning.

## Run Teacher Distillation

```bash
LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 \
DISTILL_OUT=outputs/teacher_distill/hard64_lumina_qwen \
bash scripts/distill/run_teacher_distill.sh
```

The command resumes existing JSONL outputs by `sample_id`. Failed samples are
written to `errors.jsonl`. The default prompt mode is compact JSON-only output,
which is required for `bailian/qwen3.7-plus`. If a response is non-JSON, the
teacher makes one text-only JSON repair attempt by default. Error rows keep a
short `raw_preview` for diagnosis, never full raw text unless
`--include-raw-text` is explicitly set. If Qwen returns free-form reasoning and
the follow-up text-only repair call comes back empty, the distiller emits a
conservative abstention localization label with fallback metadata and prunes the
resolved sample from `errors.jsonl` on the next successful rerun.

## Audit, Export, And Baseline

```bash
python -m ascr.distill.audit \
  --distill-dir outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact

python -m ascr.distill.export_dataset \
  --distill-dir outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact \
  --output outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl

python -m ascr.training.train_selector \
  --task cell-prior \
  --dataset outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
  --output-dir outputs/stage2_baselines/cell_prior_qwen37
```

`cell-prior` is a lightweight baseline only. It does not implement the Stage-2
learned selector model or DDP training.

## Cluster Split

On the current HKU cluster, login nodes can reach the OFOX endpoint but compute
nodes cannot resolve or route to `api.ofox.ai`. Run API teacher distillation on
the login node, then use compute nodes only for downstream jobs that read the
exported `dataset.jsonl`.

Do not use `jobs/distill/api_teacher_distill.sbatch` on this cluster until
compute-node DNS/egress or proxy access is fixed. The job is retained as a
template, but it is disabled by default and requires
`ASCR_ALLOW_COMPUTE_API_DISTILL=1` to run.

For compute-node baseline training after export:

```bash
sbatch --export=ALL,DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl,OUTPUT_DIR=outputs/stage2_baselines/cell_prior_qwen37_holdout,EVAL_MODE=holdout,TRAIN_RATIO=0.8,SEED=0,TOP_K=3 \
  jobs/training/stage2_cell_prior_baseline.sbatch
```

Small JSON teacher artifacts may be intentionally committed with `git add -f`.
Do not commit images, logs, model weights, caches, `.env`, or API keys.

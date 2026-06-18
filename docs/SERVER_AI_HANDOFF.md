# Server AI Handoff

This file is for an AI assistant or shell session running on the university GPU
server. The local repository has been hardened and pushed to GitHub. The server
copy should sync to the latest `main`, validate the local server environment,
run the requested smoke/full jobs, and record detailed results for the local
Codex session to inspect next.

Important: after you finish, append a detailed entry to `docs/AI_COLLAB_LOG.md`.
That log is the shared notebook between the local Codex session and the
server-side assistant. The human will pass your log entry back to local Codex,
so include enough detail for local Codex to understand exactly what happened
without SSH access.

## Current Git Target

- Repository: https://github.com/hmss2002/ASCR.git
- Branch: main
- Expected minimum commit: latest `origin/main`; must include `docs/API_TEACHER_DISTILL.md` and `jobs/distill/api_teacher_distill.sbatch`

## Sync The Server Checkout

If the repository is already present:

```bash
cd ASCR
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
```

If the repository is not present:

```bash
git clone https://github.com/hmss2002/ASCR.git
cd ASCR
git checkout main
git rev-parse HEAD
```

Confirm that `git rev-parse HEAD` matches the latest pushed `origin/main` given
by the local Codex session.

## Check Server Dependencies

Create model-family environments if they do not already exist:

```bash
python3.11 -m venv .venv-qwen36
source .venv-qwen36/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements/qwen_vl.txt
deactivate

python3.11 -m venv .venv-lumina
source .venv-lumina/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements/lumina.txt
deactivate
```

Set paths for the server's real model/cache locations:

```bash
export QWEN_MODEL_PATH=/path/to/qwen3.5-9b
export LUMINA_REPO=/path/to/Lumina-DiMOO
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

Do not create, print, or commit real API keys. If API judging is needed, read
`OFOX_API_KEY` from the shell or scheduler environment only.

For API teacher distillation, also set:

```bash
export OFOX_BASE_URL=${OFOX_BASE_URL:-https://api.ofox.ai/v1}
export ASCR_TEACHER_MODEL=${ASCR_TEACHER_MODEL:-bailian/qwen3.7-plus}
export ASCR_TEACHER_QUALITY_MAX_TOKENS=${ASCR_TEACHER_QUALITY_MAX_TOKENS:-2048}
export ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=${ASCR_TEACHER_LOCALIZATION_MAX_TOKENS:-2048}
export ASCR_TEACHER_JSON_REPAIR_RETRIES=${ASCR_TEACHER_JSON_REPAIR_RETRIES:-1}
```

## Validate

```bash
source .venv-qwen36/bin/activate
python scripts/smoke_test.py --server --skip-dry-run
python -m ascr.cli.preflight --mode server \
  --config configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml \
  --scan-secrets
```

If the model paths are not ready yet, report exactly which path is missing.

## Run Smoke Jobs

```bash
PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_lumina_qwen bash scripts/run_multigpu.sh
MODE=mmada-self PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_mmada_self bash scripts/run_multigpu.sh
```

## Run Main 8-GPU Inference

```bash
MODE=lumina-qwen-8gpu PROMPT_LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 \
  bash scripts/run_multigpu.sh
```

## Run API Teacher Distillation On The Login Node

First read `docs/API_TEACHER_DISTILL.md`. Then confirm that
`outputs/lumina_qwen_hard64/records` and matching `runs/pXXX/*/trace.jsonl`
exist. Use the real API key only through the login-node shell environment.
Do not submit OFOX/API teacher distillation to compute nodes on the current
cluster network posture; compute nodes cannot reach `api.ofox.ai`.

Probe the API:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048
export ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=2048
export ASCR_TEACHER_JSON_REPAIR_RETRIES=1
source .venv-qwen36/bin/activate
python scripts/distill/api_probe.py --allow-empty-content
```

Run or resume the 64-sample teacher pass. If this directory already contains
the previous compact run, the command skips successful labels and retries only
unresolved tasks such as p037 localization:

```bash
LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 \
DISTILL_OUT=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact \
bash scripts/distill/run_teacher_distill.sh
```

Audit and export the dataset:

```bash
python -m ascr.distill.audit \
  --distill-dir outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact

python -m ascr.distill.export_dataset \
  --distill-dir outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact \
  --output outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl
```

## Run Dataset-Consuming Baselines On Compute Nodes

The current in-tree downstream training path is the API-free `cell-prior`
baseline. It consumes the exported `dataset.jsonl` and can run on compute nodes:

```bash
DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
OUTPUT_DIR=outputs/stage2_baselines/cell_prior_qwen37_holdout \
EVAL_MODE=holdout \
TRAIN_RATIO=0.8 \
SEED=0 \
TOP_K=3 \
bash scripts/training/run_cell_prior.sh
```

For batch execution, submit:

```bash
sbatch --export=ALL,DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl,OUTPUT_DIR=outputs/stage2_baselines/cell_prior_qwen37_holdout,EVAL_MODE=holdout,TRAIN_RATIO=0.8,SEED=0,TOP_K=3 \
  jobs/training/stage2_cell_prior_baseline.sbatch
```

## Run Offline Selector Benchmarks

After `cell-prior` writes `selector_prior.json`, run the API-free benchmark
harness. It evaluates the labeled holdout split as in-domain and writes
out-domain predictions for DrawBench smoke prompts as an unlabeled readiness
check:

```bash
SELECTOR=outputs/stage2_baselines/cell_prior_qwen37_holdout/selector_prior.json \
IN_DOMAIN_DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
IN_DOMAIN_SPLIT=outputs/stage2_baselines/cell_prior_qwen37_holdout/split_manifest.json \
OUT_DOMAIN_PROMPTS=configs/benchmarks/prompts/drawbench_smoke8.txt \
OUT_DOMAIN_LIMIT=8 \
OUTPUT_DIR=outputs/selector_benchmarks/cell_prior_qwen37 \
TOP_K=3 \
bash scripts/benchmark/run_selector_benchmark.sh
```

The out-domain smoke prompts do not have cell-level teacher labels yet, so the
out-domain section reports `label_status=unlabeled_prompts_only` instead of an
accuracy metric. Add teacher labels later before treating out-domain accuracy as
a real benchmark.

## Run Student Localizer Before/After Image Benchmarks

Read `docs/STUDENT_LOCALIZER_IMAGE_BENCHMARK.md` before running this section.
This is the first real distilled-student image-quality workflow. It does not
treat `cell-prior` as the student model. The student is
`grid-localizer-v0`, which predicts semantic error cells from the prompt and
current grid image; the existing `GridSemanticReopeningSelector` then maps those
cells to token reopen masks inside the normal ASCR loop.

Train the student localizer from the canonical Qwen3.7 compact teacher dataset:

```bash
source .venv-qwen36/bin/activate

python -m ascr.training.train_localizer \
  --task grid-localizer-v0 \
  --dataset outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
  --image-root outputs/lumina_qwen_hard64 \
  --output-dir outputs/stage2_students/grid_localizer_v0 \
  --eval-mode holdout \
  --train-ratio 0.8 \
  --seed 0
```

Run in-domain and Geneval-smoke image generation on GPU nodes. These jobs must
not receive `OFOX_API_KEY`.

```bash
source .venv-lumina/bin/activate

STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json \
PROMPTS=outputs/stage2_students/grid_localizer_v0/holdout_prompts.txt \
DOMAIN=in_domain_hard64_holdout \
OUTPUT_DIR=outputs/image_bench/student_localizer_v0/in_domain_hard64_holdout \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_student_image_benchmark.sh

STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json \
PROMPTS=configs/benchmarks/prompts/geneval_553.txt \
DOMAIN=geneval_smoke16 \
LIMIT=16 \
OUTPUT_DIR=outputs/image_bench/student_localizer_v0/geneval_smoke16 \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_student_image_benchmark.sh
```

For Slurm:

```bash
sbatch --export=ALL,STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json,PROMPTS=configs/benchmarks/prompts/geneval_553.txt,DOMAIN=geneval_smoke16,LIMIT=16,OUTPUT_DIR=outputs/image_bench/student_localizer_v0/geneval_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/student_image_benchmark_lumina.sbatch
```

After image generation finishes, run Qwen3.7 before/after judging on the login
node only:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048

source .venv-qwen36/bin/activate

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/student_localizer_v0/in_domain_hard64_holdout/manifest.jsonl \
  --output-dir outputs/api_judges/student_localizer_v0/in_domain_hard64_holdout \
  --keep-going

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/student_localizer_v0/geneval_smoke16/manifest.jsonl \
  --output-dir outputs/api_judges/student_localizer_v0/geneval_smoke16 \
  --keep-going
```

Append exact commands, Slurm job ids, GPU node names, generation counts, judge
winner counts, score deltas, failures, and output paths to
`docs/AI_COLLAB_LOG.md`. Commit and push safe small JSON summaries only. Do not
commit generated images, logs, model weights, caches, `.env`, or API keys.

`jobs/distill/api_teacher_distill.sbatch` is retained as a template but is
disabled by default. Do not use it until compute-node DNS/egress or proxy access
to `api.ofox.ai` is fixed and confirmed.

For this teacher/distill handoff, push the detailed update to GitHub after
validation. It is acceptable to force-add the small JSON outputs listed below,
but do not add images, logs, checkpoints, model weights, caches, `.env`, or API
keys:

```bash
git add -f outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/*.json*
git add -f outputs/stage2_baselines/cell_prior_qwen37_holdout/*.json*
git add -f outputs/selector_benchmarks/cell_prior_qwen37/*.json*
git add docs/AI_COLLAB_LOG.md
git commit -m "Run compute-node cell prior baseline"
git push
```

Current multi-GPU support is Stage-1 single-node prompt sharding with paired
worker processes. It is not `torchrun` DDP. Stage-2 training is scaffolded but
not implemented as a runnable DDP training pipeline.

## Report Back

Append a detailed entry to `docs/AI_COLLAB_LOG.md`, then report the same
information to the human. The report must include:

1. `git rev-parse HEAD`
2. `python --version`
3. `nvidia-smi` summary
4. whether preflight passed
5. any missing model/checkpoint/repo path
6. Slurm job ids and output log paths
7. exact commands run and whether each passed, failed, or was skipped
8. files changed, if any, and why
9. output directories and important result files created
10. warnings/errors that local Codex should inspect next
11. for API teacher distillation: API model name, number of API calls attempted,
    number of localization labels, number of quality labels, number of errors,
    output paths for `manifest.json`, `localization_labels.jsonl`,
    `quality_labels.jsonl`, and `errors.jsonl`

If you modify files on the server, run validation, commit the changes, and push
them to GitHub only if they are safe to sync. Never commit secrets, model
weights, checkpoints, generated outputs, local caches, Slurm logs, datasets, or
virtual environments. If you are unsure whether a server-side change should be
committed, leave it uncommitted and describe it in `docs/AI_COLLAB_LOG.md`.

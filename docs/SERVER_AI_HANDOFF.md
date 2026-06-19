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
- Expected minimum commit: latest `origin/main`; must include
  `docs/SERVER_AI_TASK_LUMINA_LORA_JSON_V3_DATA_EXPANSION.md`

## Current Priority Override

Read and follow this file for the next Stage-2 server pass:

```text
docs/SERVER_AI_TASK_LUMINA_LORA_JSON_V3_DATA_EXPANSION.md
```

The old `grid-localizer-v0/v1` workflow below is historical scaffold context.
Do not run it as the formal distilled-student benchmark. The current blocker is
Lumina-native evaluator JSON compliance: LoRA v2 learned JSON-like text but
still had `parse_rate=0.0`. The next server task is Qwen3.7 teacher-data
expansion plus LoRA v3 with clean canonical targets and all-mask answer
training.

For new server-side work, create a branch from updated `main`:

```bash
cd ASCR
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b feat/lumina-lora-json-v3-data-server
```

## Current Stage-2 Priority

The formal Stage-2 target is now **Lumina-native semantic evaluator
distillation**. Do not treat `grid-localizer-v0` or `grid-localizer-v1` as the
main distilled student. Those workflows are retained only as scaffold/sanity
baselines.

The server has already shown that `LuminaNativeEngine.answer_image()` can use
Lumina-DiMOO's MMU path to read an image and return text. The unresolved gate is
JSON compliance: the current raw output is natural language, so the evaluator
abstains instead of reopening tokens.

First rerun the audit and a JSON compliance probe:

```bash
source .venv-lumina/bin/activate

LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO} \
LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo} \
bash scripts/training/run_lumina_native_audit.sh

DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
IMAGE_ROOT=outputs/lumina_qwen_hard64 \
OUTPUT_DIR=outputs/stage2_lumina_native/json_probe \
bash scripts/training/run_lumina_native_json_probe.sh
```

Then prepare a small Qwen-teacher SFT dataset. This does not launch LoRA/SFT;
it verifies that the supervision rows are ready for the next server branch:

```bash
DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
IMAGE_ROOT=outputs/lumina_qwen_hard64 \
OUTPUT_DIR=outputs/stage2_lumina_native/sft_smoke \
LIMIT=16 \
bash scripts/training/prepare_lumina_native_sft.sh
```

Decision rule:

- If JSON probe parse rate is poor, do not run formal before/after benchmark.
  Inspect Lumina-DiMOO training/MMU code and implement a LoRA/SFT smoke path.
- If JSON probe reliably returns valid `SemanticEvaluation` JSON, run the
  Lumina-native before/after benchmark smoke and then login-node Qwen3.7 judge.

Append audit/probe/SFT results to `docs/AI_COLLAB_LOG.md` with exact commands,
job ids, GPU/node, parse counts, output paths, branch, and commit hash.

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

## Run Formal Lumina-Native Before/After Benchmarks

Run this section only after JSON compliance is acceptable or after a
Lumina-native SFT smoke adapter is available. This is the formal Stage-2 image
workflow:

```text
before: Lumina direct generation
after:  Lumina-native evaluator + GridSemanticReopeningSelector + ASCR loop
```

Compute/GPU jobs must not receive `OFOX_API_KEY`:

```bash
source .venv-lumina/bin/activate

PROMPTS=configs/benchmarks/prompts/t2i_compbench_hard64.txt \
DOMAIN=in_domain_hard64_smoke16 \
LIMIT=16 \
OUTPUT_DIR=outputs/image_bench/lumina_native/in_domain_hard64_smoke16 \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_lumina_native_image_benchmark.sh

PROMPTS=configs/benchmarks/prompts/geneval_553.txt \
DOMAIN=geneval_smoke16 \
LIMIT=16 \
OUTPUT_DIR=outputs/image_bench/lumina_native/geneval_smoke16 \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_lumina_native_image_benchmark.sh
```

For Slurm:

```bash
sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,PROMPTS=configs/benchmarks/prompts/t2i_compbench_hard64.txt,DOMAIN=in_domain_hard64_smoke16,LIMIT=16,OUTPUT_DIR=outputs/image_bench/lumina_native/in_domain_hard64_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/lumina_native_image_benchmark.sbatch

sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,PROMPTS=configs/benchmarks/prompts/geneval_553.txt,DOMAIN=geneval_smoke16,LIMIT=16,OUTPUT_DIR=outputs/image_bench/lumina_native/geneval_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/lumina_native_image_benchmark.sbatch
```

Then run Qwen3.7 before/after judging on the login node only:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048

source .venv-qwen36/bin/activate

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/lumina_native/in_domain_hard64_smoke16/manifest.jsonl \
  --output-dir outputs/api_judges/lumina_native/in_domain_hard64_smoke16 \
  --keep-going

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/lumina_native/geneval_smoke16/manifest.jsonl \
  --output-dir outputs/api_judges/lumina_native/geneval_smoke16 \
  --keep-going
```

Append exact commands, Slurm job ids, GPU node names, generation counts, judge
winner counts, score deltas, failures, and output paths to
`docs/AI_COLLAB_LOG.md`. Commit and push safe small JSON summaries only. Do not
commit generated images, logs, model weights, caches, `.env`, or API keys.

When inspecting benchmark manifests, check whether any row has
`fallback_applied=true`. The manifest's `after_image` represents the actual
last candidate image, while `selected_after_image` records the conservative
fallback-selected image when `return_initial_on_max_error` fires.

## Historical Scaffold: Student Localizer v0

Read `docs/STUDENT_LOCALIZER_IMAGE_BENCHMARK.md` before running this section.
This external student-localizer workflow is retained for reproducing earlier
scaffold results only. It is not the formal Stage-2 distilled student.

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
sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json,PROMPTS=outputs/stage2_students/grid_localizer_v0/holdout_prompts.txt,DOMAIN=in_domain_hard64_holdout,OUTPUT_DIR=outputs/image_bench/student_localizer_v0/in_domain_hard64_holdout,MAX_ITERATIONS=3 \
  jobs/benchmarks/student_image_benchmark_lumina.sbatch

sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json,PROMPTS=configs/benchmarks/prompts/geneval_553.txt,DOMAIN=geneval_smoke16,LIMIT=16,OUTPUT_DIR=outputs/image_bench/student_localizer_v0/geneval_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/student_image_benchmark_lumina.sbatch
```

Use the same shared Slurm wrapper for both image-generation paths. Do not let
compute-node image benchmark jobs inherit live OFOX/API judge variables from the
login shell; blank them explicitly in `--export` as shown above. The wrapper
also strips any leaked OFOX/API judge variables defensively.

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

When inspecting benchmark manifests, check whether any row has
`fallback_applied=true`. The manifest's `after_image` now represents the actual
last candidate image, while `selected_after_image` records the conservative
fallback-selected image when `return_initial_on_max_error` fires.

## Historical Scaffold: Student Localizer v1

This section is retained for reproducibility of the earlier scaffold baseline.
It is not the current formal Stage-2 path. Do not run it as the main next step
unless the human explicitly asks to revisit the external localizer scaffold.

The current v0 baseline is scientifically useful but weak: the latest server
run recorded `grid-localizer-v0` holdout hit_any `0.2`, in-domain judge delta
about `+0.0118`, and Geneval smoke all ties. Do not rerun v0 unchanged. The
next step is to add teacher localization labels for the existing v0 image
benchmark manifests, merge them into a v1 training dataset, train
`grid-localizer-v1`, and rerun the same before/after benchmark.

Login-node teacher localization:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=2048
export ASCR_TEACHER_JSON_REPAIR_RETRIES=1

source .venv-qwen36/bin/activate

python -m ascr.distill.localize_image_manifest \
  --manifest outputs/image_bench/student_localizer_v0/in_domain_hard64_holdout/manifest.jsonl \
  --output-dir outputs/teacher_distill/student_localizer_v1/in_domain \
  --image-fields before_grid_image,after_grid_image \
  --keep-going

python -m ascr.distill.localize_image_manifest \
  --manifest outputs/image_bench/student_localizer_v0/geneval_smoke16/manifest.jsonl \
  --output-dir outputs/teacher_distill/student_localizer_v1/geneval_smoke16 \
  --image-fields before_grid_image,after_grid_image \
  --keep-going

python -m ascr.distill.export_localizer_dataset \
  --base-dataset outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
  --extra-localizations outputs/teacher_distill/student_localizer_v1/in_domain/localization_labels.jsonl \
  --extra-localizations outputs/teacher_distill/student_localizer_v1/geneval_smoke16/localization_labels.jsonl \
  --output outputs/teacher_distill/student_localizer_v1/dataset.jsonl
```

Train v1:

```bash
python -m ascr.training.train_localizer \
  --task grid-localizer-v1 \
  --dataset outputs/teacher_distill/student_localizer_v1/dataset.jsonl \
  --image-root . \
  --output-dir outputs/stage2_students/grid_localizer_v1 \
  --eval-mode holdout \
  --train-ratio 0.8 \
  --seed 0
```

Run v1 image generation on GPU nodes. Keep OFOX/API variables blanked:

```bash
sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,STUDENT_MODEL=outputs/stage2_students/grid_localizer_v1/student_model.json,PROMPTS=outputs/stage2_students/grid_localizer_v1/holdout_prompts.txt,DOMAIN=in_domain_hard64_holdout_v1,OUTPUT_DIR=outputs/image_bench/student_localizer_v1/in_domain_hard64_holdout,MAX_ITERATIONS=3 \
  jobs/benchmarks/student_image_benchmark_lumina.sbatch

sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,STUDENT_MODEL=outputs/stage2_students/grid_localizer_v1/student_model.json,PROMPTS=configs/benchmarks/prompts/geneval_553.txt,DOMAIN=geneval_smoke16_v1,LIMIT=16,OUTPUT_DIR=outputs/image_bench/student_localizer_v1/geneval_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/student_image_benchmark_lumina.sbatch
```

Judge v1 on the login node and compare to v0:

```bash
python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/student_localizer_v1/in_domain_hard64_holdout/manifest.jsonl \
  --output-dir outputs/api_judges/student_localizer_v1/in_domain_hard64_holdout \
  --keep-going --overwrite

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/student_localizer_v1/geneval_smoke16/manifest.jsonl \
  --output-dir outputs/api_judges/student_localizer_v1/geneval_smoke16 \
  --keep-going --overwrite

python -m ascr.benchmarks.compare_image_judges \
  --baseline-summary outputs/api_judges/student_localizer_v0/in_domain_hard64_holdout/summary.json \
  --candidate-summary outputs/api_judges/student_localizer_v1/in_domain_hard64_holdout/summary.json \
  --output outputs/api_judges/student_localizer_v1/in_domain_hard64_holdout/compare_to_v0.json

python -m ascr.benchmarks.compare_image_judges \
  --baseline-summary outputs/api_judges/student_localizer_v0/geneval_smoke16/summary.json \
  --candidate-summary outputs/api_judges/student_localizer_v1/geneval_smoke16/summary.json \
  --output outputs/api_judges/student_localizer_v1/geneval_smoke16/compare_to_v0.json
```

Append the full v1 results to `docs/AI_COLLAB_LOG.md`, including API label
counts, Slurm job ids, v1 metrics, judge summaries, compare-to-v0 reports, and
any warnings. Push safe small JSON outputs only:

```bash
git add docs/AI_COLLAB_LOG.md
git add -f outputs/teacher_distill/student_localizer_v1/**/*.json*
git add -f outputs/stage2_students/grid_localizer_v1/*.json*
git add -f outputs/api_judges/student_localizer_v1/**/*.json*
git commit -m "Run student localizer v1 benchmark"
git push
```

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

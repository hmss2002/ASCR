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

## Run API Teacher Distillation

First read `docs/API_TEACHER_DISTILL.md`. Then confirm that
`outputs/lumina_qwen_hard64/records` and matching `runs/pXXX/*/trace.jsonl`
exist. Use the real API key only through the shell or Slurm environment.

Probe the API:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
source .venv-qwen36/bin/activate
python scripts/distill/api_probe.py
```

Run the first 64-sample teacher pass:

```bash
LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 \
DISTILL_OUT=outputs/teacher_distill/hard64_lumina_qwen \
bash scripts/distill/run_teacher_distill.sh
```

If policy requires batch execution, submit:

```bash
sbatch --export=ALL,OFOX_API_KEY,ASCR_TEACHER_MODEL=bailian/qwen3.7-plus,LIMIT=64,OUT_ROOT=outputs/lumina_qwen_hard64 \
  jobs/distill/api_teacher_distill.sbatch
```

If the compute node cannot reach the API, record the failure in
`docs/AI_COLLAB_LOG.md` and rerun the non-Slurm command on the login node if
that is permitted by server policy.

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

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
- Expected minimum commit: 4a12e963911c4bb05a3853da22627288703bfb85

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

Confirm that `git rev-parse HEAD` is `4a12e963911c4bb05a3853da22627288703bfb85`
or newer.

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

If you modify files on the server, run validation, commit the changes, and push
them to GitHub only if they are safe to sync. Never commit secrets, model
weights, checkpoints, generated outputs, local caches, Slurm logs, datasets, or
virtual environments. If you are unsure whether a server-side change should be
committed, leave it uncommitted and describe it in `docs/AI_COLLAB_LOG.md`.

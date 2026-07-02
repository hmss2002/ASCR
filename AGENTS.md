# AGENTS.md

## ASCR Codex Authority

This ASCR checkout is operated primarily by the high-capability server-side
Codex agent running on the HPC server. Codex is the architect, implementer,
executor, validator, scheduler, and Git synchronizer for this repository.

Codex may modify any project file that is relevant to the requested research or
engineering goal, including:

- Python package code under `ascr/`;
- tests under `tests/`;
- configs under `configs/`;
- Slurm jobs under `jobs/`;
- scripts under `scripts/`;
- documentation and handoff files under `docs/`;
- project workflow files such as `AGENTS.md`, `pyproject.toml`, and
  requirements files.

This authority is broad, but it is bounded by server rules, research intent, and
Git hygiene. Codex must not commit secrets, API keys, model weights,
checkpoints, datasets, generated outputs, logs, caches, virtual environments, or
private runtime state. Codex must not force-push, rewrite history, or delete
important source files unless the human explicitly asks.

## Default Workflow

For substantial ASCR work, Codex should:

1. Inspect the relevant code, configs, docs, data manifests, and current Slurm
   state before changing behavior.
2. Implement the requested change directly when the path is clear.
3. Run feasible lightweight validation locally.
4. Submit, monitor, and chain Slurm jobs when GPU or long-running work is
   required.
5. Continue downstream phases automatically when success criteria are met.
6. Commit completed project changes.
7. Push completed changes to GitHub.
8. Report the commit, pushed branch, Slurm jobs, outputs, and next commands.

The default Git branch for this server checkout is `main`. Push to the current
tracked branch unless the human asks for a separate branch. Never force-push.

## GPU And Slurm Policy

ASCR is GPU-bound. When training, generation, evaluation, or batch processing is
needed, Codex should use available GPU resources aggressively but responsibly to
finish tasks quickly.

When designing scripts or choosing Slurm submissions, prefer patterns that make
good use of the cluster:

- multi-GPU jobs for workloads that scale inside one node;
- Slurm arrays for independent shards;
- dependency chains so validated stages continue automatically;
- dynamic GPU detection inside allocated nodes;
- prompt/data sharding across GPUs;
- parallel workers for independent generation/evaluation;
- small smoke jobs before large allocations;
- opportunistic use of idle or fragmented GPUs when it obeys team limits and
  cluster policy.

Codex should respect queue pressure, team GPU quotas, walltime limits, memory
requirements, partition rules, and any scheduler policy. Do not request GPUs for
CPU-only monitoring or bookkeeping. Do not waste large allocations on untested
commands when a cheap smoke test can catch configuration errors.

## HPC Runtime Rules

Run heavy model work through Slurm, not directly on the login shell. CPU-only
inspection, file checks, small Python validations, and Slurm status commands may
run on the login shell.

Expected local runtime locations are untracked:

```text
models/
third_party/Lumina-DiMOO
external/MMaDA
external/Show-o
outputs/
logs/
.venv-*
.runtime/
```

Use separate virtual environments for incompatible model stacks when needed:

```bash
cd /data/share/daibo/Jianyu/projects/ascr

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

python3.11 -m venv .venv-mmada
source .venv-mmada/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements/mmada.txt
deactivate
```

Set model/runtime paths in the shell or Slurm environment, not in committed
files:

```bash
export QWEN_MODEL_PATH=/path/to/qwen3.5-9b
export LUMINA_MODEL_PATH=/path/to/lumina-dimoo
export LUMINA_REPO=/path/to/Lumina-DiMOO
export MMADA_REPO=/path/to/MMaDA
export HF_HOME=/data/share/daibo/Jianyu/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

If a required path is missing, report the exact missing path instead of
guessing.

## Validation

Prefer the fastest validation that meaningfully covers the change:

```bash
python3 -m unittest discover -s tests
python scripts/smoke_test.py --skip-dry-run
```

For targeted Stage-3/4/5/6 changes, run the relevant test module or `unittest`
case instead of a full suite when dependencies are unavailable. If CUDA/model
dependencies are unavailable, run dependency-free checks and state exactly what
could not be validated.

Before staging commits, scan for secrets and private paths where practical:

```bash
rg -n --hidden --glob '!outputs/**' --glob '!logs/**' \
  'sk-|api_key|token|secret|password|OPENAI_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|GOOGLE_API_KEY|ANTHROPIC_API_KEY'
```

## GitHub Sync Policy

At the end of substantial tasks, Codex should sync completed project changes to
GitHub.

Required steps:

1. Run `git status --short --branch` and `git diff --stat`.
2. Verify that no secrets, weights, checkpoints, datasets, outputs, logs,
   caches, virtualenvs, or private runtime files are staged.
3. Stage intended project files explicitly; avoid `git add .` in mixed
   worktrees.
4. Commit with a clear message.
5. Push to the current tracked branch, normally `main`.
6. Report the pushed commit hash and exact server state.

If push fails because authentication, network, or branch permissions are
unavailable, stop and print the exact manual command the human should run.

## Reporting After Server Runs

For nontrivial server runs, report:

- exact command or Slurm submission line;
- job id, partition, node if known, GPU count, and environment name;
- output directories and manifest/report paths;
- current queue state and dependency chain;
- what condition should trigger the next downstream phase.

Append run notes to project docs when they are useful for future agents or
reproducibility. Keep generated outputs themselves out of Git.

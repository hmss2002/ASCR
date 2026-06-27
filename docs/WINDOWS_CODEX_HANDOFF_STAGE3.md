# Windows Codex Handoff: ASCR Latest Stage-3 State

This document is for a Codex agent running on Windows. Its job is to pull the
latest ASCR repository, understand the most recent local Mac work, and continue
without duplicating or confusing Stage 2 and Stage 3.

## 1. Task Understanding

You are helping migrate the latest ASCR project state from the user's Mac/GitHub
workflow to Windows Codex.

The project is:

```text
ASCR: Alternating Semantic-Confidence Revision
GitHub: https://github.com/hmss2002/ASCR.git
Latest known pushed commit: check `git log -1 --oneline --decorate`
Commit message: may advance as Windows and server agents alternate
Main working branch: main
```

Your first goal on Windows is not to redesign the project. Your first goal is to
clone or update the repository, install the lightweight local development
environment, verify the latest code, and read the handoff documents.

## 2. Recommended Windows Location

Use a normal coding directory, not Downloads or Desktop:

```powershell
mkdir $HOME\Projects -Force
cd $HOME\Projects
```

Expected final location:

```text
C:\Users\<username>\Projects\ASCR
```

If the repository already exists there, update it instead of recloning.

## 3. Clone Or Pull Latest Code

### If ASCR does not exist yet

```powershell
cd $HOME\Projects
git clone https://github.com/hmss2002/ASCR.git
cd ASCR
git checkout main
git pull --ff-only
```

### If ASCR already exists

```powershell
cd $HOME\Projects\ASCR
git status --short --branch
git fetch origin
git checkout main
git pull --ff-only
```

After syncing, verify:

```powershell
git log -1 --oneline --decorate
git rev-parse HEAD
git rev-parse origin/main
```

If `HEAD` and `origin/main` differ, report the exact output before making
changes. Also check recent remote feature branches because the server AI often
pushes results to a new branch before `main` is updated:

```powershell
git fetch --all --prune
git for-each-ref --sort=-committerdate --format="%(committerdate:iso8601) %(refname:short) %(objectname:short) %(subject)" refs/remotes/origin
```

## 4. Python Environment On Windows

Prefer a project-local virtual environment. Do not install packages into system
Python or Anaconda base.

The repo includes Windows helper scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/bootstrap_local.ps1
powershell -ExecutionPolicy Bypass -File scripts/setup/activate_local.ps1
```

If the helper script fails, use `uv` directly:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

If `uv` is not installed, install it using the official method or ask the user
before choosing a package manager. Do not use random third-party installers.

## 5. Lightweight Verification

Run the same checks that passed on the Mac:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vq_corruptor tests.test_token_locality tests.test_schema_parser
.\.venv\Scripts\python.exe -m ascr.cli.token_locality_probe --help
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

Expected:

```text
Focused tests: OK
token_locality_probe --help: prints CLI options without loading Lumina
scripts/smoke_test.py: all tests pass, mock dry-run passes, secret scan passes
```

Warnings about missing CUDA, missing Lumina weights, missing Qwen weights, or
missing `third_party/Lumina-DiMOO` are expected on a local Windows machine. Heavy
Lumina/Qwen work belongs on the university GPU server.

## 6. What Was Just Changed On The Mac

The Mac Codex reviewed the existing ASCR architecture and made Stage 3 explicit
without disrupting Stage 1 or Stage 2.

Stage 1 remains:

```text
zero-training ASCR loop:
prompt -> generator -> evaluator JSON -> selector -> TokenReopenMask -> reopen
```

Stage 2 remains:

```text
Lumina-native SemanticEvaluation JSON distillation
docs/LUMINA_NATIVE_DISTILLATION.md
```

New Stage 3 is:

```text
self-corrupted token repair:
clean Lumina vq_ids
-> controlled token corruption
-> clean/corrupted image pair
-> known corruption mask as self-supervised label
-> locality probe
-> later selector / internal repair head
```

Important new files:

```text
docs/STAGE3_SELF_CORRUPTED_TOKEN_REPAIR.md
docs/SERVER_AI_TASK_STAGE3_SELF_CORRUPT_LOCALITY.md
ascr/corruption/vq_corruptor.py
ascr/analysis/token_locality.py
ascr/cli/token_locality_probe.py
configs/stage3/self_corrupt/locality_probe_smoke.yaml
jobs/stage3/self_corrupt_locality_probe.sbatch
tests/test_vq_corruptor.py
tests/test_token_locality.py
```

Also updated:

```text
README.md
docs/AI_COLLAB_LOG.md
pyproject.toml
ascr/core/schemas.py
tests/test_schema_parser.py
```

`GridCell.from_any()` now accepts labels beyond 4x4, for example `H8` and
`P16`, so Stage 3 can compare 8x8 and 16x16 selector grids.

## 7. Current Stage-3 State

The server AI ran the first Stage-3 gate on branch
`feat/stage3-self-corrupt-locality-server`:

```text
token-to-image locality probe for controlled Lumina VQ-token corruption
```

Result summary:

- job 71441 completed successfully on one GPU;
- 8 prompts and 24 corruption rows decoded successfully;
- `block_4x4_random_replace` and `local_shuffle_4x4` showed clear locality on
  4x4 and 8x8 grids;
- top-1 and top-k hit rates were 1.00 across tested grids and corruption types;
- Phase 2 dataset construction is now the next step.

The next server task document is:

```text
docs/SERVER_AI_TASK_STAGE3_SELF_CORRUPT_DATASET.md
```

Expected server commands after pulling latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
source .venv-lumina/bin/activate

python -m ascr.cli.stage3_locality_report \
  --manifest outputs/stage3_self_corrupt/locality_probe_smoke/manifest.jsonl \
  --summary outputs/stage3_self_corrupt/locality_probe_smoke/summary.json \
  --output-dir outputs/stage3_self_corrupt/locality_probe_smoke/report

python -m ascr.cli.stage3_self_corrupt_dataset \
  --manifest outputs/stage3_self_corrupt/locality_probe_smoke/manifest.jsonl \
  --summary outputs/stage3_self_corrupt/locality_probe_smoke/summary.json \
  --output-dir outputs/stage3_self_corrupt/datasets/locality_smoke_v1
```

Server AI should append results to `docs/AI_COLLAB_LOG.md`. It should not yet
train selectors, inspect hidden states, use Qwen/Gemini teacher labels, or run
formal before/after ASCR benchmarks.

## 8. Windows Codex Behavior Rules

Before making edits:

```powershell
git status --short --branch
git pull --ff-only
```

Do not change unrelated files.

Do not commit:

- `.venv`;
- `outputs`;
- `logs`;
- model weights;
- checkpoints;
- local caches;
- API keys or credentials.

For substantial changes, run:

```powershell
git diff --check
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

Before committing, scan status and diff:

```powershell
git status --short --branch
git diff --stat
git diff --cached --stat
```

Use clear commit messages, for example:

```text
Add stage3 self-corruption dataset builder
```

Push normally:

```powershell
git push
```

Never force-push unless the user explicitly asks.

## 9. Current Recommended Next Local Work

Windows Codex can safely do documentation and pure-Python work. Good next tasks:

1. Read `docs/STAGE3_SELF_CORRUPTED_TOKEN_REPAIR.md`.
2. Read `docs/AI_COLLAB_LOG.md` from the latest Stage-3 entry.
3. Verify tests pass locally.
4. Use the server locality probe result to build the Phase-2 dataset before
   implementing selector training or hidden-state repair-head work.

If the user wants local progress before server results, implement only
model-light utilities such as:

- dataset JSONL schema validation;
- aggregation scripts for locality `manifest.jsonl`;
- unit tests for metrics;
- documentation updates.

Do not write GPU-only Lumina training code that cannot be validated at least
syntactically or with pure-Python tests.

## 10. Short Human Summary

The Mac side has already pushed the latest Stage-3 direction and locality probe
scaffold to GitHub. Windows Codex should pull `main`, verify the local dev
environment, read the Stage-3 and server-task documents, and avoid diverging
from the server workflow. The next scientific gate is not more planning; it is
the server-side locality probe.

# Server Setup

This document describes the intended Linux/Slurm setup for ASCR after cloning the
repository on a GPU server. Keep model weights, third-party checkouts, generated
outputs, logs, and secrets out of Git.

## Environment Layout

Use separate virtual environments for model families whose dependency stacks are
not compatible:

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

python3.11 -m venv .venv-mmada
source .venv-mmada/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements/mmada.txt
deactivate
```

Override the names with `ASCR_ENV_QWEN`, `ASCR_ENV_LUMINA`, and
`ASCR_ENV_MMADA` if your cluster stores environments elsewhere.

## Runtime Paths

Expected untracked runtime paths:

```text
models/qwen3.5-9b
models/lumina-dimoo
models/mmada-8b-mixcot
models/show-o-512x512
models/magvitv2
models/phi-1_5
third_party/Lumina-DiMOO
external/MMaDA
external/Show-o
```

Run this before submitting jobs:

```bash
python -m ascr.cli.preflight \
  --mode server \
  --config configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml \
  --scan-secrets
```

You can also run the bundled smoke helper:

```bash
source .venv-qwen36/bin/activate
python scripts/smoke_test.py --server --skip-dry-run
```

## Smoke Jobs

Use tiny prompt limits first:

```bash
PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_lumina_qwen \
  sbatch jobs/smoke/stage1_lumina_qwen_2gpu.sbatch

PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_mmada_self \
  sbatch jobs/smoke/stage1_mmada_self_1gpu.sbatch
```

Then submit the full sharded jobs. Stage-1 multi-GPU inference is prompt sharding
across resident worker processes. It is not DDP or `torchrun`.

```bash
PROMPT_LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 \
  sbatch jobs/stage1/lumina/stage1_lumina_qwen_coarse_hard64_8gpu.sbatch
```

The same supported jobs can be submitted through a small wrapper:

```bash
PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_lumina_qwen bash scripts/run_multigpu.sh
MODE=mmada-self PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_mmada_self bash scripts/run_multigpu.sh
MODE=lumina-qwen-8gpu PROMPT_LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 bash scripts/run_multigpu.sh
```

For multi-node runs, set `NODE_INDEX` and `NODE_COUNT` per submitted node so the
global prompt shard assignment remains disjoint.

For a command-oriented checklist, see `docs/REMOTE_RUN.md`.

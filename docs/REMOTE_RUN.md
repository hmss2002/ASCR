# Remote Run Guide

This is the short path for running ASCR after cloning on a Linux GPU server.
Keep model weights, caches, `.env`, logs, and generated outputs outside Git.

## 1. Clone Or Update

Fresh clone:

```bash
git clone https://github.com/hmss2002/ASCR.git
cd ASCR
git checkout main
```

Existing checkout:

```bash
cd ASCR
git fetch origin
git checkout main
git pull --ff-only origin main
```

## 2. Create Environments

Use separate environments for incompatible model stacks:

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

## 3. Set Runtime Paths

Adjust these to the server layout:

```bash
export QWEN_MODEL_PATH=/path/to/qwen3.5-9b
export LUMINA_REPO=/path/to/Lumina-DiMOO
export MMADA_REPO=/path/to/MMaDA
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

For API judges, set keys only in the shell or scheduler environment:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048
export ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=2048
export ASCR_TEACHER_JSON_REPAIR_RETRIES=1
```

Never write real keys into tracked files.

## 4. Smoke Test

```bash
source .venv-qwen36/bin/activate
python scripts/smoke_test.py --server --skip-dry-run
```

If CUDA/model paths are not ready yet, run the local-only version:

```bash
python scripts/smoke_test.py --skip-dry-run
```

## 5. Single-GPU Or CPU Mock Run

The single-process wrapper is useful for dry runs and debugging one prompt:

```bash
DRY_RUN=1 OUT_ROOT=outputs/local_dry_run bash scripts/run_inference.sh
```

For real single-process inference, activate the matching model environment and
set `GENERATOR`, `EVALUATOR`, `CONFIG`, and model paths explicitly:

```bash
source .venv-lumina/bin/activate
GENERATOR=lumina EVALUATOR=qwen_vl MAX_ITERS=1 \
CONFIG=configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml \
PROMPT="A red cube left of a blue sphere" \
OUT_ROOT=outputs/single_prompt \
bash scripts/run_inference.sh
```

Lumina + Qwen usually requires the paired two-environment IPC jobs below because
their dependency stacks are intentionally separated.

## 6. Multi-GPU Slurm Runs

Two-GPU Lumina/Qwen smoke:

```bash
PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_lumina_qwen \
  bash scripts/run_multigpu.sh
```

One-GPU MMaDA self smoke:

```bash
MODE=mmada-self PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_mmada_self \
  bash scripts/run_multigpu.sh
```

Eight-GPU Lumina/Qwen Stage-1 shard run:

```bash
MODE=lumina-qwen-8gpu PROMPT_LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 \
  bash scripts/run_multigpu.sh
```

Equivalent generic Slurm wrapper:

```bash
MODE=lumina-qwen PROMPT_LIMIT=1 OUT_ROOT=outputs/smoke_lumina_qwen \
  sbatch scripts/slurm_infer.sbatch
```

For the generic wrapper's 8-GPU mode, override Slurm resources at submission
time because `#SBATCH` lines are read before shell variables:

```bash
MODE=lumina-qwen-8gpu PROMPT_LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 \
  sbatch --gres=gpu:8 --cpus-per-task=64 --mem=400G --time=08:00:00 scripts/slurm_infer.sbatch
```

## Supported Parallelism

Current Stage-1 multi-GPU support is single-node prompt sharding with paired
worker processes. It is not `torchrun` DDP. Multi-node can be coordinated by
setting `NODE_INDEX` and `NODE_COUNT` per node/job so prompt shards are disjoint.
Stage-2 training code is reserved and should not be treated as implemented DDP
training until `ascr.training.train_selector` is filled in.

## 7. API Teacher Distillation

After `outputs/lumina_qwen_hard64` exists, generate teacher labels with:

```bash
python scripts/distill/api_probe.py --allow-empty-content
LIMIT=64 OUT_ROOT=outputs/lumina_qwen_hard64 bash scripts/distill/run_teacher_distill.sh
```

Then audit, export, and run the lightweight baseline:

```bash
python -m ascr.distill.audit --distill-dir outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact
python -m ascr.distill.export_dataset \
  --distill-dir outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact \
  --output outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl
python -m ascr.training.train_selector \
  --task cell-prior \
  --dataset outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
  --output-dir outputs/stage2_baselines/cell_prior_qwen37
```

For Slurm:

```bash
sbatch --export=ALL,DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl,OUTPUT_DIR=outputs/stage2_baselines/cell_prior_qwen37_holdout,EVAL_MODE=holdout,TRAIN_RATIO=0.8,SEED=0,TOP_K=3 \
  jobs/training/stage2_cell_prior_baseline.sbatch
```

Offline selector benchmark:

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

See `docs/API_TEACHER_DISTILL.md` for schema and troubleshooting.

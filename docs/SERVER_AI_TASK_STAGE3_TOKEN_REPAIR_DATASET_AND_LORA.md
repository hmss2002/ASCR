# Server AI Task: Stage3 Token Repair Dataset And Stage4 LoRA

Status: ready for server execution after pulling the latest GitHub commit from Windows Codex.

This task replaces the older multi-grid locality branch as the current Stage3 mainline. Do not redesign the label schema unless the data proves it is broken.

## Shared Collaboration Contract

- Shared log: `docs/AI_COLLAB_LOG.md`.
- Server AI should append a dated entry after every run: commands, commit hash, job ids, output paths, metrics, failures, and exact fixes.
- Windows Codex owns most local code edits, script wiring, and difficult architecture/debug analysis.
- Server AI owns GPU execution, log inspection, simple server-side fixes, and small safe patches when the issue is obvious.
- If Server AI edits code, commit and push its own Git branch. Do not leave server-only fixes unpushed.
- If a branch is not `main`, write the branch name, commit hash, and pull command in `docs/AI_COLLAB_LOG.md`.

The cluster has more than one hundred GPUs across many nodes, with eight GPUs per node. The project philosophy is to request enough resources early so each runnable task finishes quickly. Prefer parallel data-generation jobs over long serial jobs when outputs are shard-safe.

## Core Research Logic

We assume the 64x64 Lumina discrete VQ token state already contains the information needed for this Stage3 localization task. Therefore labels come directly from token-level corruption masks, not from decoded images or human image inspection.

Canonical dataset row:

```json
{"cells": ["D4", "D5"]}
```

Negative row:

```json
{"cells": []}
```

Fixed choices:

- Input to model: original prompt plus corrupted 64x64 VQ tokens.
- Output schema: `{"cells": string[]}`.
- Action grid: exactly 8x8.
- Ground truth: corrupted 64x64 token mask projected to 8x8 cells.
- Positive data: clean tokens -> choose token mask -> apply corruption operator -> corrupted tokens -> project token mask to 8x8 cells.
- Negative data: same clean tokens with no corruption.
- Decoded images: optional audit only, never used for labels.

Supported mask sizes:

- `1x1`
- `2x2`
- `4x4`
- `8x8`

Supported value operators:

- `random_replace`
- `local_shuffle`
- `neighbor_copy`
- `transplant`

`mask_resample` is intentionally excluded.

The 8x8 label has no boundary ambiguity. Each 64x64 token belongs to exactly one 8x8 cell, and a corruption block that crosses a cell boundary returns every intersected cell.

## In-Context Prompt Policy

The in-context prompt is only a formatting and behavior constraint for asking the MMU where the token error is. It should teach the model that:

- the task is token-state repair localization;
- the repair grid is fixed 8x8;
- it must output compact JSON only;
- no-error cases are valid;
- cells are labels such as `D4`.

Expected prompt style is implemented in `ascr.training.stage4_mmu_lora.mmu_localization_prompt(..., target_schema="repair_cells")`:

```text
You are the ASCR token-state repair cell selector.

Input: the original text prompt plus the current generated image represented as Lumina VQ tokens.

Task: choose which cells on the fixed 8x8 repair grid should be reopened because they contain corrupted VQ tokens.

Return exactly one compact JSON object and nothing else.

Schema:
{"cells": string[]}

Examples:
{"cells":["D4","D5"]}
{"cells":["A1"]}
{"cells":["A8","B8"]}
{"cells":["C3","C4","D3","D4"]}
{"cells":[]}

Rules:
- Use only 8x8 cell labels: A1 through H8.
- If no repair is needed, return {"cells":[]}.
- If corrupted tokens touch multiple cells, include every touched cell.
- Sort cells row-major: A1,A2,...,A8,B1,...,H8.
- Do not output any key except "cells".
- Do not output markdown, prose, confidence, coordinates, explanations, or extra fields.
- Use at most 8 cells.

Original prompt:
...
```

Do not ask the model to output legacy keys like `error`, `has_error`, or `corrupted_cells_8x8` for this mainline. The local parser still accepts old `{"error":...,"cells":[...]}` rows for compatibility, but new datasets and LoRA targets must be cells-only.

## Resource Strategy

Data generation is shard-safe and should use as much parallelism as the cluster allows:

1. Best: one multi-node Slurm job, each node uses 8 GPUs.
2. Fallback: a Slurm job array, one node per array task, each node uses 8 GPUs.
3. Last fallback: one node, 8 GPUs.

LoRA training is not shard-merge-safe. Do not train several independent LoRA adapters and concatenate them. Current training path is one authoritative DDP job on one 8-GPU node using `jobs/stage4/train_mmu_lora_ddp.sbatch`.

## Commands

Clone or update:

```bash
git clone <repo-url> ASCR || true
cd ASCR
git fetch origin
git checkout <branch-from-windows-codex>
git pull --ff-only
```

Environment:

```bash
python -m venv .venv-lumina
source .venv-lumina/bin/activate
python -m pip install -U pip
python -m pip install -e ".[lumina]"
python -m pip install bitsandbytes datasets
export LUMINA_REPO=third_party/Lumina-DiMOO
export LUMINA_MODEL_PATH=models/lumina-dimoo
```

Review the plan:

```bash
MODE=plan bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=plan bash scripts/training/run_stage4_token_repair_lora.sh
```

Prompt preparation is now preseeded in Git. Pulling `main` gives:

- `configs/benchmarks/prompts/diffusiondb_10k.jsonl`
- `configs/benchmarks/prompts/diffusiondb_10k.txt`
- `configs/benchmarks/prompts/stage3_token_repair_prompts_10k.txt`

The server should normally skip DiffusionDB download and just verify/reuse the files:

```bash
MODE=download_prompts bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=sample_prompts bash scripts/training/run_stage3_token_repair_dataset.sh
```

Those commands are idempotent: if the Git-tracked prompt files already have 10k rows,
they print a reuse message. To deliberately refresh prompts, run:

```bash
FORCE_DOWNLOAD=1 MODE=download_prompts bash scripts/training/run_stage3_token_repair_dataset.sh
FORCE_RESAMPLE=1 MODE=sample_prompts bash scripts/training/run_stage3_token_repair_dataset.sh
```

Preferred clean-token generation on the current H200 server:

```bash
MODE=submit_clean_h200 bash scripts/training/run_stage3_token_repair_dataset.sh
```

This submits 32 short array tasks with 4 GPUs each. The current account limit
allows 16 GPUs concurrently, so this keeps the allocation full while finished
tasks hand off to pending tasks.

For explicit speed/quality experiments, lower Lumina clean-token sampling steps:

```bash
GENERATION_TIMESTEPS=32 MODE=submit_clean_h200 \
  bash scripts/training/run_stage3_token_repair_dataset.sh
```

Default remains `GENERATION_TIMESTEPS=64`; the manifest records timesteps,
guidance scale, and temperature for each clean-token row.

Full H200 pipeline submit, including dependent merge/build/SFT/train/probe:

```bash
bash scripts/training/run_h200_token_repair_pipeline.sh
```

Fallback clean-token generation, job array where each task uses one 8-GPU node:

```bash
MODE=submit_clean PROMPTS_PER_TASK=1024 \
  bash scripts/training/run_stage3_token_repair_dataset.sh
```

After clean jobs finish:

```bash
MODE=merge_clean bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=report_clean REPORT_MIN_ROWS=10000 bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=build_dataset bash scripts/training/run_stage3_token_repair_dataset.sh
```

`MODE=report_clean` writes a Stage3 clean-token report under the clean-token
output root and fails on critical issues: too few rows, duplicate sample IDs,
duplicate prompt indexes, malformed JSONL, or missing referenced token files.
Unmanifested token files are reported so stale or still-running shard outputs
are visible before building the dataset.

Optional audit decode:

```bash
MODE=audit_decode AUDIT_PAIRS=16 bash scripts/training/run_stage3_token_repair_dataset.sh
```

Prepare Lumina SFT:

```bash
MODE=prepare_sft bash scripts/training/run_stage4_token_repair_lora.sh
MODE=convert_sft bash scripts/training/run_stage4_token_repair_lora.sh
```

Optional zero-shot probe:

```bash
MODE=probe_zero bash scripts/training/run_stage4_token_repair_lora.sh
```

Train LoRA on one 8-GPU node:

```bash
MODE=submit_train bash scripts/training/run_stage4_token_repair_lora.sh
```

Training progress visibility is enabled by default in the token-repair train
config:

```yaml
progress_bar: true
progress_every_steps: 50
validation_every_epochs: 3
```

Rank 0 prints flushed per-N-step progress in DDP runs, so Slurm logs should no
longer stay silent until epoch end. Existing jobs submitted before this change
will not gain the new logging retroactively; resubmit if live progress is
needed. To override without editing YAML:

```bash
PROGRESS_EVERY_STEPS=10 PROGRESS_BAR=1 VALIDATION_EVERY_EPOCHS=1 MODE=submit_train \
  bash scripts/training/run_stage4_token_repair_lora.sh
```

Probe trained LoRA:

```bash
MODE=probe_lora bash scripts/training/run_stage4_token_repair_lora.sh
```

Summarize measured training speed after `training_manifest.json` exists:

```bash
MODE=speed_report bash scripts/training/run_stage4_token_repair_lora.sh
```

If a previous L40S `training_manifest.json` is available, compare directly:

```bash
SPEED_BASELINE_MANIFESTS=outputs/path/to/l40s/training_manifest.json \
SPEED_BASELINE_LABELS=l40s_1024_gc \
SPEED_BASELINE_LABEL=l40s_1024_gc \
MODE=speed_report bash scripts/training/run_stage4_token_repair_lora.sh
```

## Expected Outputs

- Raw DiffusionDB prompts: `configs/benchmarks/prompts/diffusiondb_10k.txt`
- Training prompts: `configs/benchmarks/prompts/stage3_token_repair_prompts_10k.txt`
- Clean token manifest: `outputs/stage3_token_repair/clean_tokens_h200_32x4/clean_manifest.jsonl`
- Clean token report: `outputs/stage3_token_repair/clean_tokens_h200_32x4/report/stage3_clean_manifest_report.json`
- Dataset: `outputs/stage3_token_repair/datasets/repair_cells_40k/dataset.jsonl`
- Dataset manifest: `outputs/stage3_token_repair/datasets/repair_cells_40k/dataset_manifest.json`
- SFT examples: `outputs/stage4_token_repair/repair_cells_8x8/sft/`
- Lumina SFT JSONL: `outputs/stage4_token_repair/repair_cells_8x8/lumina_sft/`
- LoRA adapter: `outputs/stage4_token_repair/repair_cells_8x8/lora_h200_1024px_adamw`
- Speed report: `outputs/stage4_token_repair/repair_cells_8x8/speed_report/stage4_speed_report.json`
- Probe summary: `outputs/stage4_token_repair/repair_cells_8x8/probe_lora_h200_1024px_eval/summary.json`

Do not commit outputs, datasets, model weights, adapter checkpoints, prompt downloads, logs, or cache directories.

## Success Criteria

- Dataset has 30k positive rows and 10k negative rows.
- Clean-token report passes with at least 10k rows and no missing token files or duplicate clean samples.
- Every target JSON has exactly one key, `cells`.
- Positive rows have nonempty cells.
- Negative rows have empty cells.
- SFT split is group-safe by `source_clean_sample_id`, so clean and corrupted variants of the same prompt do not cross train/val/test.
- LoRA probe parse rate should be near 1.0 before interpreting localization metrics.

## If Something Fails

- If DiffusionDB download fails, use the Git-tracked `diffusiondb_10k` / `stage3_token_repair_prompts_10k` files. They were downloaded locally specifically to avoid server disk-quota and streaming failures.
- If multi-node clean generation fails, switch to the job-array fallback.
- If single-node 8-GPU LoRA training OOMs, inspect the latest DDP log first. Do not start many independent LoRA jobs; that does not produce one mergeable adapter.
- If parser output reverts to legacy keys or misses the required `cells` key, rerun with `target_schema=repair_cells` and inspect the exact prompt in `sft_examples.jsonl`.

# Server AI Task: Stage 3 Selector Baselines

You are the server-side AI for ASCR. The Stage-3 Phase-2 self-corruption
dataset is ready. Your task is to run Phase-3 selector baselines over the
existing dataset and report the metrics.

## Context

Read first:

```bash
docs/STAGE3_SELF_CORRUPTED_TOKEN_REPAIR.md
docs/AI_COLLAB_LOG.md
```

Current dataset from the previous server task:

```text
outputs/stage3_self_corrupt/datasets/locality_smoke_v1/dataset.jsonl
outputs/stage3_self_corrupt/datasets/locality_smoke_v1/dataset_manifest.json
```

The dataset has 24 rows: 8 prompts x 3 corruption types. All referenced image
and token paths were verified on the server.

## Branch

Create a server branch from latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
git checkout -b feat/stage3-self-corrupt-selectors-server
```

## Run

The selector baselines are model-light and do not load Lumina. They can run on
the login node, but an sbatch wrapper is also provided for reproducible cluster
logs.

Direct command:

```bash
source .venv-lumina/bin/activate
python -m ascr.cli.stage3_train_selectors \
  --config configs/stage3/self_corrupt/selector_baselines_smoke.yaml
```

Shell wrapper:

```bash
source .venv-lumina/bin/activate
bash scripts/training/run_stage3_selector_baselines.sh
```

Slurm wrapper:

```bash
sbatch jobs/stage3/train_self_corrupt_selectors.sbatch
```

Default output root:

```text
outputs/stage3_self_corrupt/selectors/locality_smoke_v1/
```

Expected outputs:

```text
summary.json
grid4/split_manifest.json
grid4/random/{selector_model.json,metrics.json,predictions.jsonl}
grid4/token_prior/{selector_model.json,metrics.json,predictions.jsonl}
grid4/rgb_diff_oracle/{selector_model.json,metrics.json,predictions.jsonl}
grid4/rgb_localizer/{selector_model.json,metrics.json,predictions.jsonl}
grid4/prompt_rgb_localizer/{selector_model.json,metrics.json,predictions.jsonl}
grid8/...
grid16/...
```

## Baselines

Run all configured baselines:

- `random`: deterministic random cell selector.
- `token_prior`: trains a cell-frequency prior from the training split.
- `rgb_diff_oracle`: uses clean-vs-corrupted image differences; this is an
  oracle upper bound, not a deployable selector.
- `rgb_localizer`: trains a small per-cell RGB-feature logistic localizer on
  corrupted images.
- `prompt_rgb_localizer`: same as `rgb_localizer`, with hashed prompt features.

Default grids are 4x4, 8x8, and 16x16. The split is a stratified 75/25 holdout
by corruption type.

## Report Back In `docs/AI_COLLAB_LOG.md`

Append a dated server entry with:

- branch name and commit hash;
- host, environment path, and exact command or Slurm job id;
- output root;
- for each grid and baseline: hit_any_rate, mean_f1_at_k, mean_iou, and
  mean_distance_to_target_cells from `summary.json`;
- any missing image/token paths or runtime blockers;
- recommendation for the next Stage-3 step.

Do not commit generated `outputs/` files. Commit only the log update.

## Decision Rule

- If token-prior is close to the learned localizers, the current 24-row smoke
  dataset is too small for model conclusions. Expand the self-corruption
  dataset before neural selector work.
- If `rgb_localizer` or `prompt_rgb_localizer` beats random and token-prior on
  holdout at 4x4/8x8, expand the dataset and keep the coarse-to-fine selector
  path.
- Treat `rgb_diff_oracle` as an upper bound only.

## Do Not Do Yet

- Do not inspect Lumina hidden states.
- Do not add an internal repair head yet.
- Do not run formal before/after ASCR image benchmarks yet.
- Do not use Qwen/Gemini teacher labels for Stage-3 training labels.

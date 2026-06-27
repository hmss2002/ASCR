# Server AI Task: Stage 3 Self-Corruption Dataset Build

Status: completed by server branch `feat/stage3-self-corrupt-dataset-server`.
Keep this file as provenance. The next server task is
`docs/SERVER_AI_TASK_STAGE3_SELF_CORRUPT_SELECTORS.md`.

You are the server-side AI for ASCR. The Stage-3 locality smoke has passed, so
your next task is to convert the existing locality probe output into a
repeatable report and Phase-2 self-corruption dataset.

## Context

Read first:

```bash
docs/STAGE3_SELF_CORRUPTED_TOKEN_REPAIR.md
docs/AI_COLLAB_LOG.md
```

Important result from job 71441:

- 8 prompts, 24 corruption rows, 72 locality heatmaps.
- Lumina generation and decode succeeded for all clean/corrupted examples.
- `block_4x4_random_replace` and `local_shuffle_4x4` show clear locality on
  4x4 and 8x8 grids.
- Proceed to Phase 2 dataset construction.

## Branch

Create a server branch from latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
git checkout -b feat/stage3-self-corrupt-dataset-server
```

## Inputs

Use the existing locality smoke output from job 71441:

```text
outputs/stage3_self_corrupt/locality_probe_smoke/
  manifest.jsonl
  summary.json
  heatmaps/
  images/
  tokens/
```

Do not rerun Lumina unless those files are missing.

## Run

Activate the environment:

```bash
source .venv-lumina/bin/activate
```

Create the aggregate locality report:

```bash
python -m ascr.cli.stage3_locality_report \
  --manifest outputs/stage3_self_corrupt/locality_probe_smoke/manifest.jsonl \
  --summary outputs/stage3_self_corrupt/locality_probe_smoke/summary.json \
  --output-dir outputs/stage3_self_corrupt/locality_probe_smoke/report
```

Create the Phase-2 dataset:

```bash
python -m ascr.cli.stage3_self_corrupt_dataset \
  --manifest outputs/stage3_self_corrupt/locality_probe_smoke/manifest.jsonl \
  --summary outputs/stage3_self_corrupt/locality_probe_smoke/summary.json \
  --output-dir outputs/stage3_self_corrupt/datasets/locality_smoke_v1
```

Expected outputs:

```text
outputs/stage3_self_corrupt/locality_probe_smoke/report/locality_report.json
outputs/stage3_self_corrupt/locality_probe_smoke/report/locality_report.md
outputs/stage3_self_corrupt/datasets/locality_smoke_v1/dataset.jsonl
outputs/stage3_self_corrupt/datasets/locality_smoke_v1/dataset_manifest.json
```

## Report Back In `docs/AI_COLLAB_LOG.md`

Append a dated server entry with:

- branch name and commit hash;
- host, environment path, and exact commands;
- report output paths;
- dataset output paths;
- dataset row count and corruption types;
- whether all referenced clean/corrupted token and image paths exist;
- blockers, if any.

Do not commit the generated `outputs/` files. Commit only the log update.

## Do Not Do Yet

- Do not train a selector yet.
- Do not inspect hidden states or add a repair head yet.
- Do not run formal before/after ASCR image benchmarks yet.
- Do not use Qwen/Gemini teacher labels for Stage-3 training labels.

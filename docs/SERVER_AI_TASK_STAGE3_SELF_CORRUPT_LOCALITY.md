# Server AI Task: Stage 3 Self-Corruption Locality Probe

You are the server-side AI for ASCR. Your immediate task is to run the first
Stage-3 gate: token-to-image locality for controlled Lumina VQ-token corruption.

## Context

Local Codex changed the research roadmap:

- Stage 2 remains Lumina-native `SemanticEvaluation JSON` distillation.
- New Stage 3 is self-corrupted token repair.
- The first question is whether corrupting a local set of Lumina VQ tokens causes
  decoded-image changes mostly near the corresponding spatial region.

Read first:

```bash
docs/STAGE3_SELF_CORRUPTED_TOKEN_REPAIR.md
```

Implemented local tooling:

```text
ascr/corruption/vq_corruptor.py
ascr/analysis/token_locality.py
ascr/cli/token_locality_probe.py
configs/stage3/self_corrupt/locality_probe_smoke.yaml
jobs/stage3/self_corrupt_locality_probe.sbatch
```

## Branch

Create a server branch from latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
git checkout -b feat/stage3-self-corrupt-locality-server
```

## Environment

Use the Lumina environment on a GPU node:

```bash
source .venv-lumina/bin/activate
export LUMINA_REPO=third_party/Lumina-DiMOO
export LUMINA_MODEL_PATH=models/lumina-dimoo
```

Do not export OFOX/API teacher keys for this task. This is not a teacher-label
job.

## Run

Start with the smoke job:

```bash
sbatch jobs/stage3/self_corrupt_locality_probe.sbatch
```

Or run directly on an allocated GPU:

```bash
python -m ascr.cli.token_locality_probe \
  --config configs/stage3/self_corrupt/locality_probe_smoke.yaml
```

Expected output root:

```text
outputs/stage3_self_corrupt/locality_probe_smoke/
```

## Report Back In `docs/AI_COLLAB_LOG.md`

Append a dated server entry with:

- branch name and commit hash;
- host, GPU node, Slurm job id, and environment path;
- exact command;
- whether model loading/generation/decode succeeded;
- row count and prompt count from `summary.json`;
- per-corruption aggregate notes from `manifest.jsonl`;
- example heatmap paths;
- observed blocker, if any.

Minimum metrics to report for each analysis grid and corruption type:

- mean `inside_energy_fraction`;
- mean `inside_outside_energy_ratio`, ignoring null values;
- mean `center_displacement_cells`, ignoring null values;
- top-1 hit rate;
- top-k hit rate;
- median `effective_radius_cells`.

## Decision Rule

If 2x2 or 4x4 random replacement has clear locality on 8x8 or 16x16 analysis
grids, recommend proceeding to self-corruption dataset construction.

If only 4x4 block corruption is stable, recommend coarse-to-fine Stage 3 rather
than direct 64x64 token selection.

If locality is weak or decode fails for corrupted ids, stop and report the exact
failure before attempting selector training.

## Do Not Do Yet

- Do not train a selector.
- Do not run hidden-state repair-head experiments.
- Do not use Qwen/Gemini teacher labels.
- Do not run formal before/after ASCR image benchmarks.

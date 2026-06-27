# Server AI Task: Stage 4 Hidden-State Repair Head

You are the server-side AI for ASCR. Phase 3 cleared on the 128-row Hard64
self-corruption dataset. Your task is to start Phase 4 by checking Lumina hidden
state access, extracting hidden features, and training the first lightweight
repair head.

## Context

Read first:

```bash
docs/STAGE3_SELF_CORRUPTED_TOKEN_REPAIR.md
docs/AI_COLLAB_LOG.md
```

Current evidence:

- Hard64 self-corruption dataset: 128 rows, 64 prompts x 2 corruption types.
- Dataset path:
  `outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl`
- Phase 3 gate cleared: `prompt_rgb_localizer` reached 0.875 hit_any at 16x16.
- This justifies Phase 4 hidden-state probing.

## Branch

Create a server branch from latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
git checkout -b feat/stage4-hidden-repair-server
```

## Step 1: Hidden-State Capability Probe

Run first:

```bash
source .venv-lumina/bin/activate
python -m ascr.cli.stage4_hidden_state_probe \
  --config configs/stage4/self_corrupt/hidden_probe_hard64.yaml
```

Or with Slurm:

```bash
sbatch jobs/stage4/hidden_state_probe.sbatch
```

Expected output:

```text
outputs/stage4_self_corrupt/hidden_probe_hard64/hidden_state_probe.json
```

Decision gate:

- If `supports_hidden_states` is true, continue to Step 2.
- If it is false, stop and report exact failure. Do not patch Lumina internals
  blindly.

## Step 2: Extract Hidden Features

Run:

```bash
python -m ascr.cli.stage4_extract_hidden_features \
  --config configs/stage4/self_corrupt/hidden_features_hard64_grid16.yaml
```

Expected output:

```text
outputs/stage4_self_corrupt/hidden_features_hard64_grid16/hidden_features.jsonl
outputs/stage4_self_corrupt/hidden_features_hard64_grid16/hidden_features_manifest.json
```

This extracts projected hidden features per 16x16 selector cell from corrupted
image-token prompts. The default projection is 128 dimensions from the last
hidden layer.

## Step 3: Train Lightweight Repair Head

Run:

```bash
python -m ascr.cli.stage4_train_repair_head \
  --config configs/stage4/self_corrupt/repair_head_hard64_grid16.yaml
```

Or run Step 2 and Step 3 together:

```bash
sbatch jobs/stage4/train_repair_head.sbatch
```

Expected output:

```text
outputs/stage4_self_corrupt/repair_head_hard64_grid16/repair_head.json
outputs/stage4_self_corrupt/repair_head_hard64_grid16/metrics.json
outputs/stage4_self_corrupt/repair_head_hard64_grid16/predictions.jsonl
outputs/stage4_self_corrupt/repair_head_hard64_grid16/split_manifest.json
```

## Report Back In `docs/AI_COLLAB_LOG.md`

Append a dated server entry with:

- branch name and commit hash;
- host, GPU node, environment path, and exact commands/job ids;
- hidden-state probe status and hidden-state shapes;
- feature extraction row count, feature dim, hidden layer, and failure count;
- repair-head metrics: hit_any_rate, mean_f1_at_k, mean_iou;
- blockers, if any;
- recommendation for the next Stage-4 step.

Do not commit generated `outputs/` files. Commit only the log update.

## Do Not Do Yet

- Do not fine-tune Lumina.
- Do not add LoRA to hidden-state paths yet.
- Do not run formal before/after ASCR benchmarks yet.
- Do not use Qwen/Gemini labels for Stage-4 training labels.

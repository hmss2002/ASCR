# Server AI Task: Stage 4 Native MMU/LoRA Localization

You are the server-side AI for ASCR. Phase 3 cleared on the 128-row Hard64
self-corruption dataset. Start Phase 4 by using Lumina's native MMU path rather
than an external repair head.

## Context

Read first:

```bash
docs/STAGE3_SELF_CORRUPTED_TOKEN_REPAIR.md
docs/STAGE4_PROMPT_SCALING_GUIDE.md
docs/AI_COLLAB_LOG.md
```

Current evidence:

- Dataset: `outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl`
- Rows: 128, from 64 prompts and 2 corruption types.
- Phase-3 selector gate cleared: `prompt_rgb_localizer` reached `hit_any=0.875`
  at 16x16.
- Main Phase-4 principle: use Lumina MMU/UMM internally. Prefer direct
  corrupted VQ-token input via `answer_vq_tokens()` over decode/re-encode
  where possible.

## Branch

Create a server branch from latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
git checkout -b feat/stage4-mmu-lora-server
```

## Run

Activate the Lumina environment:

```bash
source .venv-lumina/bin/activate
export LUMINA_REPO=${LUMINA_REPO:-third_party/Lumina-DiMOO}
export LUMINA_MODEL_PATH=${LUMINA_MODEL_PATH:-models/lumina-dimoo}
```

Preferred single command:

```bash
bash scripts/training/run_stage4_mmu_lora.sh
```

Slurm equivalent:

```bash
sbatch jobs/stage4/train_mmu_lora.sbatch
```

Schema-aligned dual-path route after the 2026-06-28 local update:

```bash
# Full-capacity config: 1024 image tokens, full LoRA target module set.
bash scripts/training/run_stage4_mmu_lora_dual.sh

# L40S fallback matching the previous successful memory profile.
PROFILE=l40s bash scripts/training/run_stage4_mmu_lora_dual.sh
```

`PROFILE=l40s` is a memory fallback for schema-learning and parse-rate recovery.
It uses 512-token/image cropping and two LoRA target modules, so use the full
1024 config whenever bf16 + gradient checkpointing fits for final localization
metrics.

Parallel Slurm array route:

```bash
PROFILE=l40s sbatch jobs/stage4/train_mmu_lora_dual.sbatch

# After both array tasks finish, generate the comparison report.
python -m ascr.cli.stage4_compare_input_modes \
  --vq-tokens-probe outputs/stage4_self_corrupt/mmu_lora_hard64_dual/vq_tokens/probe_lora_l40s_eval/summary.json \
  --decoded-image-probe outputs/stage4_self_corrupt/mmu_lora_hard64_dual/decoded_image/probe_lora_l40s_eval/summary.json \
  --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64_dual/input_mode_comparison_l40s
```

Important schema note:

- Stage-4 self-corruption localization now trains the compact
  `localization_cells` target schema:
  `{"has_error": boolean, "corrupted_cells_4x4": [], "corrupted_cells_8x8": [], "corrupted_cells_16x16": []}`.
- The probe and `LuminaNativeEvaluator` normalize this schema back into ASCR
  `SemanticEvaluation` internally before scoring or selector/reopen use.
- The probe still accepts legacy `SemanticEvaluation` outputs and also makes a
  best-effort recovery from the old bad pattern where cell-like integers were
  emitted in `correction_instruction`.

Manual route:

```bash
python -m ascr.cli.stage4_mmu_localization_probe \
  --config configs/stage4/self_corrupt/mmu_probe_zero_hard64.yaml

python -m ascr.cli.stage4_prepare_mmu_sft \
  --config configs/stage4/self_corrupt/mmu_sft_hard64.yaml

python -m ascr.training.prepare_lumina_sft_data \
  --sft-examples outputs/stage4_self_corrupt/mmu_lora_hard64/sft/train_sft_examples.jsonl \
  --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64/lumina_sft \
  --repo-path "$LUMINA_REPO" \
  --checkpoint-path "$LUMINA_MODEL_PATH" \
  --image-size 1024

python -m ascr.cli.stage4_train_mmu_lora \
  --config configs/stage4/self_corrupt/mmu_lora_train_hard64.yaml

python -m ascr.cli.stage4_mmu_localization_probe \
  --config configs/stage4/self_corrupt/mmu_probe_lora_hard64.yaml
```

Optional ASCR smoke after LoRA evaluation:

```bash
python -m ascr.benchmarks.lumina_native_benchmark \
  --prompts configs/benchmarks/prompts/t2i_compbench_hard64.txt \
  --domain hard64_mmu_lora_smoke \
  --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64/ascr_smoke \
  --config configs/stage4/self_corrupt/mmu_lora_ascr_smoke.yaml \
  --limit 4 \
  --max-iterations 1 \
  --keep-going
```

## Expected Outputs

```text
outputs/stage4_self_corrupt/mmu_lora_hard64/
  probe_zero_sample16/{summary.json,probe_rows.jsonl,predictions.jsonl}
  sft/{manifest.json,split_manifest.json,train_sft_examples.jsonl,eval_sft_examples.jsonl}
  lumina_sft/{manifest.json,train.jsonl,image_tokens/}
  lora/{adapter_config.json,adapter_model.safetensors,training_manifest.json}
  probe_lora_eval/{summary.json,probe_rows.jsonl,predictions.jsonl}
```

Do not commit generated outputs or adapter weights.

Dual-path outputs are written under:

```text
outputs/stage4_self_corrupt/mmu_lora_hard64_dual/
  vq_tokens/{sft,lumina_sft,lora,lora_l40s,probe_zero_sample16,probe_lora_eval,probe_lora_l40s_eval}
  decoded_image/{sft,lumina_sft,lora,lora_l40s,probe_zero_sample16,probe_lora_eval,probe_lora_l40s_eval}
  input_mode_comparison*/{comparison.json,comparison.md}
```

## Report Back In `docs/AI_COLLAB_LOG.md`

Append a dated server entry with:

- branch name and commit hash;
- host, GPU node, environment path, exact commands or Slurm job ids;
- zero-shot MMU probe status: parse rate, hit_any_rate, mean_f1_at_k, mean_iou;
- SFT data counts: total/train/eval, missing image count, missing VQ-token count;
- LoRA training status: epochs, final loss, output adapter path;
- LoRA probe metrics: parse rate, hit_any_rate, mean_f1_at_k, mean_iou;
- whether direct VQ-token MMU input worked;
- blockers, if any;
- recommendation for the next Phase-4/Phase-5 step.
- for the dual-path run, include both `vq_tokens` and `decoded_image` metrics,
  the winner from `comparison.md`, and whether either path should become the
  Phase-5 default.

Commit only `docs/AI_COLLAB_LOG.md` and push the feature branch.

## Decision Gates

- If zero-shot MMU parse rate and hit_any are already useful, try prompt-only
  improvement before relying on LoRA.
- If zero-shot output is descriptive or malformed, use the LoRA path.
- If LoRA parse rate is below 0.5, simplify the target JSON format before
  scaling.
- If LoRA approaches or beats the Phase-3 16x16 external baseline
  (`prompt_rgb_localizer hit_any=0.875`), proceed to Phase-5 ASCR-loop
  integration and real prompt-following transfer evaluation.

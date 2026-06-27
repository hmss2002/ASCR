# Stage 3: Self-Corrupted Token Repair

This document is the current Stage-3 working plan. It replaces the older
"cross-model ASCR framework" Stage-3 idea as the main research direction. The
old cross-model idea remains useful later as validation after Lumina-native
self-corruption works.

## Research Question

Can controlled corruption in Lumina-DiMOO's discrete image-token space create a
self-supervised signal for localized repair, and can that signal transfer from
synthetic token corruption to real prompt-following errors?

The conservative claim is:

```text
Self-supervised token corruption improves local error detection and selective
token resampling efficiency.
```

Do not claim that the model becomes generally smarter without external signal.
The clean image is only a relative positive compared with its corrupted pair; it
is not guaranteed to be a perfect prompt match.

## Clarification: Corruption Target And Selector Grids

Stage 3 corrupts Lumina's generated VQ image-token sequence, then decodes the
corrupted token sequence back into an image. In the current Lumina path this is
a 64x64 token grid for 1024x1024 images, not a 4x4 or 8x8 selector grid.

The corruption is controlled and local:

- random replacement of one local block of token ids, for example 2x2 or 4x4;
- local shuffling inside a block, for example 4x4;
- later, smaller perturbations can be added if they produce visible but not
  destructive changes.

The goal is not to create arbitrary bad images. The goal is to create paired
clean/corrupted examples where the true corrupted token positions are known, the
decoded image quality or local consistency is measurably worse, and the
localization target is self-supervised.

The 4x4, 8x8, and 16x16 grids are selector or analysis grids inherited from the
earlier ASCR interface. They are projections of the underlying token grid. The
server locality smoke supports using coarse-to-fine selector grids: start with
4x4 or 8x8 localization, then refine only if the data supports finer claims.

## Current Repo Facts

- `LuminaAdapter` stores generated VQ ids in
  `GenerationState.metadata["vq_ids"]`.
- `LuminaNativeEngine.decode_to()` decodes VQ ids to an image.
- `LuminaNativeEngine.reopen()` copies baseline VQ ids, masks selected token
  positions with `MASK_TOKEN_ID`, and resamples only those positions.
- `DirectTokenReopeningSelector` already supports selector grids smaller than
  the token grid when the sizes divide evenly.
- Stage 2 remains the JSON evaluator path:

```text
prompt + image -> Lumina-native evaluator -> SemanticEvaluation JSON
-> GridSemanticReopeningSelector -> Lumina reopen
```

Stage 3 adds a separate direct mask path:

```text
prompt + clean/corrupted vq_ids -> self-corruption labels
-> selector or Lumina-native MMU localizer -> TokenReopenMask -> Lumina reopen
```

## Phase 0: Local Direction Reset

Local Codex should maintain:

- this design document;
- `docs/SERVER_AI_TASK_STAGE3_SELF_CORRUPT_LOCALITY.md`;
- `docs/AI_COLLAB_LOG.md` entries recording what was changed locally and what
  the server AI should run next.

The README should stay concise. It should point to this document rather than
becoming another long project-control file.

## Phase 1: Token Locality Probe

Before training, verify whether token-space corruption has mostly local visual
effects after VQ decoding.

Implemented local tooling:

- `ascr/corruption/vq_corruptor.py`
- `ascr/analysis/token_locality.py`
- `ascr/cli/token_locality_probe.py`
- `configs/stage3/self_corrupt/locality_probe_smoke.yaml`
- `jobs/stage3/self_corrupt_locality_probe.sbatch`

Default corruption operators:

- `single_random_replace`
- `block_2x2_random_replace`
- `block_4x4_random_replace`
- `local_shuffle_2x2`
- `local_shuffle_4x4`

The smoke config starts with 2x2 and 4x4 block corruption because single-token
changes may be too weak visually.

Metrics:

- center displacement in analysis-grid cells;
- inside/outside energy ratio;
- inside energy fraction;
- top-1 hit;
- top-k hit;
- effective radius for 80 percent of visual difference energy.

Server command:

```bash
sbatch jobs/stage3/self_corrupt_locality_probe.sbatch
```

Direct command:

```bash
python -m ascr.cli.token_locality_probe \
  --config configs/stage3/self_corrupt/locality_probe_smoke.yaml
```

Expected outputs:

```text
outputs/stage3_self_corrupt/locality_probe_smoke/
  manifest.jsonl
  summary.json
  heatmaps/
  images/
  tokens/
```

Decision gate:

- If 2x2 or 4x4 corruption is local, proceed to self-corruption dataset
  construction.
- If only coarse block locality is stable, use coarse-to-fine block repair
  rather than token-level claims.
- If locality is weak, do not train internal Lumina localizers yet; narrow the
  claim.

Server result from job 71441:

- 8 prompts and 24 corruption rows completed successfully.
- Lumina generation/decode succeeded for all clean and corrupted token grids.
- `block_4x4_random_replace` and `local_shuffle_4x4` show clear locality on
  4x4 and 8x8 grids.
- Top-1 and top-k hit rates were 1.00 across all tested corruption types and
  grid sizes.
- Proceed to Phase 2 dataset construction before selector training.

## Phase 2: Self-Corruption Dataset

After Phase 1 passes, build paired examples:

```text
clean_vq_ids -> corrupted_vq_ids
clean image  -> corrupted image
known selected token mask
```

Each row should include:

```json
{
  "sample_id": "p0000_c000",
  "prompt": "...",
  "clean_vq_ids_path": "...",
  "corrupted_vq_ids_path": "...",
  "corruption_indices": [[12, 23], [12, 24]],
  "corruption_type": "block_2x2_random_replace",
  "clean_image": "...",
  "corrupted_image": "...",
  "coarse_labels_4x4": ["B2"],
  "coarse_labels_8x8": ["D4"],
  "coarse_labels_16x16": ["H8"],
  "token_grid_size": 64,
  "image_size": 1024
}
```

Store large token arrays in referenced JSON files rather than inlining them in
`dataset.jsonl`.

Implemented model-light tooling:

```bash
python -m ascr.cli.stage3_locality_report \
  --manifest outputs/stage3_self_corrupt/locality_probe_smoke/manifest.jsonl \
  --summary outputs/stage3_self_corrupt/locality_probe_smoke/summary.json \
  --output-dir outputs/stage3_self_corrupt/locality_probe_smoke/report

python -m ascr.cli.stage3_self_corrupt_dataset \
  --manifest outputs/stage3_self_corrupt/locality_probe_smoke/manifest.jsonl \
  --summary outputs/stage3_self_corrupt/locality_probe_smoke/summary.json \
  --output-dir outputs/stage3_self_corrupt/datasets/locality_smoke_v1
```

These commands do not load Lumina. They convert the server probe outputs into a
repeatable report and a Phase-2 dataset manifest.

## Phase 3: Selector Baselines

Train baselines before touching Lumina hidden states:

- random baseline;
- token-prior baseline;
- RGB diff oracle baseline;
- small RGB localizer;
- prompt + corrupted image localizer.

Compare selector grids:

- 4x4;
- 8x8;
- 16x16;
- 64x64 direct token grid.

Metrics:

- precision@k;
- recall@k;
- IoU;
- coarse-cell accuracy;
- mean distance to corrupted token;
- selected token count.

Only proceed to internal Lumina/MMU work if synthetic corruption labels are
learnable above trivial baselines.

Implemented model-light baseline tooling:

```bash
python -m ascr.cli.stage3_train_selectors \
  --config configs/stage3/self_corrupt/selector_baselines_smoke.yaml
```

Equivalent wrappers:

```bash
bash scripts/training/run_stage3_selector_baselines.sh
sbatch jobs/stage3/train_self_corrupt_selectors.sbatch
```

Default output:

```text
outputs/stage3_self_corrupt/selectors/locality_smoke_v1/
  summary.json
  grid4/<baseline>/{selector_model.json,metrics.json,predictions.jsonl}
  grid8/<baseline>/{selector_model.json,metrics.json,predictions.jsonl}
  grid16/<baseline>/{selector_model.json,metrics.json,predictions.jsonl}
```

Notes:

- `rgb_diff_oracle` uses clean-vs-corrupted image differences and is an upper
  bound, not a deployable selector.
- `rgb_localizer` and `prompt_rgb_localizer` train small pure-Python per-cell
  logistic models over corrupted-image features.
- The 24-row smoke dataset is enough to validate wiring, but not enough for a
  paper-level selector conclusion. If learned localizers beat random and
  token-prior on holdout, expand the self-corruption dataset before hidden-state
  native-MMU LoRA work.

Server Hard64 result:

- Manual 8-shard GPU scale-out produced a 128-row dataset
  (`locality_hard64_v1`).
- Locality remained stable: inside energy fractions were about 0.51/0.52 for
  `block_4x4_random_replace` and `local_shuffle_4x4`, with top-1/top-k at 1.0.
- Phase-3 selector gate cleared:
  `prompt_rgb_localizer` reached 0.875 hit_any on 16x16, substantially above
  random and token-prior.
- Proceed to Phase 4 native MMU/LoRA localization.

## Phase 4: Native MMU/LoRA Localization

Main research version:

```text
prompt + corrupted image tokens
-> Lumina native MMU answer path
-> structured SemanticEvaluation JSON
-> existing selector/reopen contract
```

The key principle is "unify": the localizer should live inside Lumina's own
multimodal understanding path where the image-token information is already
represented. Avoid making a separate model the mainline unless it is explicitly
used as a baseline or diagnostic.

Training order:

1. Probe zero-training Lumina MMU localization on self-corrupted Hard64 rows.
2. Prepare MMU SFT pairs from corrupted VQ tokens and canonical
   `SemanticEvaluation` targets.
3. Train a lightweight LoRA adapter on Lumina's MMU answer path.
4. Evaluate the LoRA adapter against Phase-3 external selector baselines.
5. If useful, plug the LoRA-backed evaluator into the existing ASCR loop.

Implemented native-MMU tooling:

```bash
python -m ascr.cli.stage4_mmu_localization_probe \
  --config configs/stage4/self_corrupt/mmu_probe_zero_hard64.yaml

python -m ascr.cli.stage4_prepare_mmu_sft \
  --config configs/stage4/self_corrupt/mmu_sft_hard64.yaml

python -m ascr.training.prepare_lumina_sft_data \
  --sft-examples outputs/stage4_self_corrupt/mmu_lora_hard64/sft/train_sft_examples.jsonl \
  --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64/lumina_sft \
  --repo-path third_party/Lumina-DiMOO \
  --checkpoint-path models/lumina-dimoo \
  --image-size 1024

python -m ascr.cli.stage4_train_mmu_lora \
  --config configs/stage4/self_corrupt/mmu_lora_train_hard64.yaml

python -m ascr.cli.stage4_mmu_localization_probe \
  --config configs/stage4/self_corrupt/mmu_probe_lora_hard64.yaml
```

Single-command and Slurm wrappers:

```bash
bash scripts/training/run_stage4_mmu_lora.sh
sbatch jobs/stage4/train_mmu_lora.sbatch
```

`LuminaNativeEngine.answer_vq_tokens()` allows Stage 4 to ask the MMU about
existing corrupted VQ tokens directly, without first decoding to RGB and
re-encoding through the VQ-VAE. The decoded corrupted image path is still
available as a fallback and for human inspection, but the preferred training
input is the internal token representation.

The older hidden-state repair-head scaffold remains useful only as a diagnostic
baseline. It is not the main research route unless the native MMU/LoRA path
fails.

### Phase 4 Server Results (2026-06-28)

Server branch: `feat/stage4-mmu-lora-server`. All four sub-steps ran.

| Step | Result | Detail |
|------|--------|--------|
| 4a — Zero-shot probe | parse_rate **0.0** | 16/16 malformed. Base Lumina MMU outputs natural language, not structured JSON. Confirms need for LoRA. |
| 4b — SFT prep | 96 train / 32 eval | 0 missing images, 0 missing VQ tokens. `preferred_training_input: vq_ids_path`. |
| 4c — LoRA training | loss 9.75 → **0.157** | 15 epochs, 12.5 min. adapter_model.safetensors = 16.8 MB. |
| 4d — LoRA eval | parse_rate **0.156** (5/32) | 27/32 still malformed. hit_any = **0.0**. |

**OOM workaround**: The default config (image_size=1024, 7 LoRA target modules,
max_seq_len=6144) OOMs on the 45 GB L40S during training. Reducing to
image_size=512, max_seq_len=2048, and 2 target modules (q_proj, v_proj)
resolved the OOM but limits LoRA capacity. Inference at 1024 fits fine.

**Schema mismatch found**: The LoRA outputs `{"correction_instruction": "11,12,13,..."}`
(integer coordinates) while the probe parser expects
`{"corrupted_cells_16x16": ["J10","J9"]}` (cell labels). The LoRA learned to
output structured localization data but with the wrong key name and value
format. This is a training target ↔ parser schema mismatch, not a
failure to learn.

**Decision**: Phase 4 gate not yet cleared (parse_rate 0.156 < 0.5 threshold).
But the LoRA qualitatively changed output from natural language → structured
coordinates, proving the approach is viable. Next steps:

1. **Fix schema alignment** — make SFT targets use the same JSON keys and
   cell-label format that the probe parser expects.
2. **Recover training capacity** — use bf16 mixed precision + gradient
   checkpointing to train at image_size=1024 with more target modules.
3. **Consider image-space MMU** — `answer_image()` on decoded corrupted images
   may be easier for the model than `answer_vq_tokens()` on raw VQ tokens.
4. **Coarse-first curriculum** — train on 4×4 grid localization first, then
   progress to 8×8 and 16×16.

## Phase 5: ASCR Loop Integration

Add a Stage-3 loop only after a selector is useful:

```text
Lumina generate
-> selector/internal critic predicts TokenReopenMask
-> Lumina reopen
-> optional accept/reject scorer
```

Compatibility rules:

- Do not break `ascr.cli.run_stage1`.
- Do not break `lumina_native_evaluator`.
- Stage-3 artifacts should still include trace, selected indices, selected
  token count, decoded images, and final summary.

Initial accept policy:

```text
if mask empty: stop
else: always accept reopened candidate
```

Add scoring only after the selector produces nontrivial masks.

## Phase 6: Synthetic-To-Real Transfer Evaluation

Train on synthetic corruption pairs. Test on real prompt-following errors:

- Hard64 holdout;
- Geneval smoke16;
- selected compositional prompts.

Benchmark arms:

- Lumina direct generation;
- Stage 1 Qwen coarse ASCR;
- Stage 2 Lumina-native JSON evaluator if available;
- Stage 3 self-corrupt selector ASCR;
- Stage 4 native MMU/LoRA ASCR if Phase 4 succeeds.

External Qwen/Gemini judges are allowed for evaluation. They must not provide
Stage-3 training labels unless the experiment is explicitly marked as a hybrid
ablation.

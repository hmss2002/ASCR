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
-> selector or internal repair head -> TokenReopenMask -> Lumina reopen
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
- If locality is weak, do not train a hidden-state repair head yet; narrow the
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

Only proceed to internal hidden-state work if synthetic corruption labels are
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
  repair-head work.

## Phase 4: Internal Repair Head

Main research version:

```text
prompt + corrupted image tokens
-> Lumina multimodal forward / MMU hidden states
-> lightweight repair head
-> H x W mask logits
```

Training order:

1. Freeze Lumina completely.
2. Train only a small repair head.
3. Compare against random, prior, RGB, and text-output baselines.
4. If frozen-head results are useful, try LoRA on the MMU path.

Server AI must first inspect whether
`LLaDAForMultiModalGeneration.forward(output_hidden_states=True)` works in the
actual Lumina checkout. If not, this phase should pause rather than forcing a
large invasive model patch.

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
- Stage 3 internal repair head ASCR if Phase 4 succeeds.

External Qwen/Gemini judges are allowed for evaluation. They must not provide
Stage-3 training labels unless the experiment is explicitly marked as a hybrid
ablation.

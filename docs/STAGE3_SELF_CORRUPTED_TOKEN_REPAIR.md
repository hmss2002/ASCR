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

## Shared TODO: Resolve 1024px LoRA Training Memory On 45 GB L40S

**Status**: Windows Codex + Server AI joint investigation. Not yet resolved.

**Problem**: A single L40S (45 GB) can run Lumina-DiMOO inference at 1024×1024
without issue. But LoRA training at 1024×1024 with the full 7-module target set
(q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj) and
max_seq_len=6144 OOMs because training adds:

- LoRA adapter parameters + gradients
- Adam optimizer states (momentum + variance ≈ 2× per parameter)
- Forward activations saved for backward pass
- Cross-entropy loss computation on long sequences

**Current workaround (L40S fallback config)**:
```
image_size: 1024 → 512          (center-crop, loses spatial alignment)
max_seq_len: 6144 → 2048        (truncates long outputs)
target_modules: 7 → 2           (only q_proj, v_proj — limits capacity)
```

**Goal**: Keep image_size=1024, max_seq_len=6144, and all 7 target modules
while fitting within 45 GB. The current full config attempts this with
bf16 + gradient checkpointing; if it OOMs, further optimizations are needed.

**Optimization candidates (either AI can implement)**:

| Technique | Expected saving | Risk |
|-----------|----------------|------|
| bf16 mixed precision | ~50% model weight memory | Already in full config |
| Gradient checkpointing | ~30% activation memory | Already in full config; ~20% slower |
| LoRA on attention only (4 modules: q/k/v/o) | ~40% optimizer memory vs 7 modules | May reduce localization quality |
| Activation offloading to CPU | Variable | Much slower |
| 8-bit Adam optimizer | ~50% optimizer memory | Small precision loss |
| Flash Attention 2 | ~20-30% activation memory | Needs compatible CUDA/code |
| Sequence parallelism | Splits seq_len across GPUs | Needs >1 GPU per training run |
| Smaller per-device batch with gradient accumulation | ~linear in batch size | Slower convergence |

**Decision**: The full config (bf16 + gradient checkpointing + 7 modules at
1024px) is the first attempt. If it OOMs, try 8-bit Adam next. If still OOMs,
reduce to 4 attention modules. Only fall back to 512px as last resort.

**Server finding (2026-06-28)**: `LLaDAForMultiModalGeneration` raises
`ValueError: does not support gradient checkpointing` — the Lumina-DiMOO
transformers fork doesn't implement `gradient_checkpointing_enable()`.
Gradient checkpointing is removed from the optimization candidate list until
Windows Codex or server AI patches the Lumina model code.

**Updated priority**:
1. ~~bf16 + gradient checkpointing~~ → **blocked** (model doesn't support gc)
2. **8-bit Adam optimizer** → installed. init works. **Server tested: 7 modules + adam8bit → OOM. 4 modules + adam8bit → OOM.** 8-bit Adam saves optimizer memory but model activations at 1024px still dominate.
3. **Gradient checkpointing in LLaDA model code** → **single highest-impact change.** See reference below. If gc works, 1024px + 7 modules fits on one L40S.
4. 512px fallback → current working config; only keep as last resort.

**Validation**: After any memory fix, re-run a single-epoch smoke to confirm
training loss decreases before committing to the full 15-epoch run.

**Implemented local follow-up (2026-06-28)**:

- `ascr.training.train_lumina_lora_smoke` supports
  `--optimizer {adamw,adamw8bit}`.
- `ascr.cli.stage4_train_mmu_lora` exposes common overrides such as
  `--epochs`, `--optimizer`, `--image-size`, `--max-seq-len`, and
  `--target-modules`, so server smoke runs do not need temporary YAML edits.
- Added 1024px memory-probe configs:
  - `mmu_lora_train_hard64_vq_tokens_l40s_1024px_adam8bit.yaml`
  - `mmu_lora_train_hard64_vq_tokens_l40s_1024px_attn4_adam8bit.yaml`
- Added coarse-to-fine curriculum configs and runners for grid4/grid8/grid16:
  `scripts/training/run_stage4_curriculum.sh` and
  `jobs/stage4/train_mmu_lora_curriculum.sbatch`.

### Reference: How Gradient Checkpointing Works (for patching LLaDA)

**Normal training forward pass:**
```
输入 → Layer1 → Layer2 → ... → Layer32 → 输出 → loss
        ↓        ↓              ↓
      存A1     存A2          存A32      (所有激活值都保留用于反向)
```
反向传播时每层都要自己的激活值。32 层 × 每层几十 MB = **几个 GB 激活值常驻显存**。

**With gradient checkpointing:**
```
输入 → Layer1 → Layer2 → ... → Layer32 → 输出 → loss
        ✗        ✗              ✗              (激活值全部丢弃)
```
反向传播时按需重算：
```
需要 A32 → 从 A31 重算一遍 Layer32 → 拿到 A32 → 算梯度 → 扔掉
需要 A31 → 从 A30 重算一遍 Layer31 → 拿到 A31 → 算梯度 → 扔掉
...
```
**用时间换空间**：每个 layer 跑两遍（~20% 慢），但激活值只保留 checkpoint 点之间那几个。

**显存对比：**
```
不开 gc:  激活值 35+ GB → OOM
开 gc:    激活值 5-8 GB  → 能跑！
```

**为什么 LLaDA 不支持：** HuggingFace 的 `PreTrainedModel` 提供了
`model.gradient_checkpointing_enable()` 接口，LLaDA 虽然继承了但一调就抛
`ValueError("LLaDAForMultiModalGeneration does not support gradient checkpointing")`。
原因是底层 transformer block 的 forward 没包 `torch.utils.checkpoint.checkpoint()`。

**大概要改什么（`modeling_llada.py`，参考，不保证精确）：**

```python
# 当前（不支持 gc）：
class LLaDADecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, ...):
        hidden_states = self.self_attn(hidden_states, attention_mask, ...)
        hidden_states = self.mlp(hidden_states, ...)
        return hidden_states

# 改成（支持 gc）：
class LLaDADecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, ...):
        if torch.is_grad_enabled() and self.training and self.gradient_checkpointing:
            def custom_forward(*inputs):
                # inputs = (hidden_states, attention_mask, ...)
                h = self.self_attn(inputs[0], inputs[1], ...)
                h = self.mlp(h, ...)
                return h
            return torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, ...
            )
        # fallback: normal forward
        hidden_states = self.self_attn(hidden_states, attention_mask, ...)
        hidden_states = self.mlp(hidden_states, ...)
        return hidden_states
```

关键点：
- 只包 transformer block 的内部 forward，不动 attention 和 MLP 内部
- 需要 `self.gradient_checkpointing` 标志位（从 `PreTrainedModel` 继承，
  `gradient_checkpointing_enable()` 会设它为 True）
- `use_reentrant=False` 在新版 PyTorch 中推荐（省显存但需要 PyTorch ≥ 1.11）
- 改动量可能就几十行但需要理解 LLaDA 的 attention 输入输出

**服务器验证方式（改完后）：**
```bash
python -m ascr.cli.stage4_train_mmu_lora \
  --config configs/stage4/self_corrupt/mmu_lora_train_hard64_vq_tokens_l40s_1024px_adam8bit.yaml \
  --gradient-checkpointing --epochs 1 --limit 4
```
如果不 OOM 且 loss 下降 → gc 生效，可以开 15 epoch 完整训练。

---

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

Parallel scale-out tooling:

```bash
# Submit prompt-windowed Slurm array shards.
PROMPT_FILE=configs/benchmarks/prompts/t2i_compbench_hard64.txt \
PROMPT_COUNT=256 \
PROMPTS_PER_TASK=8 \
OUTPUT_ROOT=outputs/stage3_self_corrupt/locality_probe_hard256 \
bash scripts/training/run_stage3_locality_parallel.sh

# Merge completed shards and build a Phase-2 dataset.
python -m ascr.cli.stage3_merge_probe_shards \
  --shard-dirs outputs/stage3_self_corrupt/locality_probe_hard256/shard_* \
  --output-dir outputs/stage3_self_corrupt/locality_probe_hard256

python -m ascr.cli.stage3_self_corrupt_dataset \
  --manifest outputs/stage3_self_corrupt/locality_probe_hard256/manifest.jsonl \
  --summary outputs/stage3_self_corrupt/locality_probe_hard256/summary.json \
  --output-dir outputs/stage3_self_corrupt/datasets/locality_hard256_v1
```

`token_locality_probe.py` supports `--prompt-offset` and `--prompt-limit`,
also readable from `PROMPT_OFFSET` and `PROMPT_LIMIT`, so Slurm array tasks no
longer need manually split prompt files. When prompt windowing is active, the
config-level `limit` is ignored unless `--limit` or `LIMIT` is explicitly set.

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
-> compact localization_cells JSON
-> normalized SemanticEvaluation
-> existing selector/reopen contract
```

The key principle is "unify": the localizer should live inside Lumina's own
multimodal understanding path where the image-token information is already
represented. Avoid making a separate model the mainline unless it is explicitly
used as a baseline or diagnostic.

Training order:

1. Probe zero-training Lumina MMU localization on self-corrupted Hard64 rows.
2. Prepare MMU SFT pairs from corrupted VQ tokens and compact
   `localization_cells` targets, then normalize them into `SemanticEvaluation`
   internally for ASCR scoring and selector/reopen integration.
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
3. **Coarse-first curriculum** — train on 4×4 grid localization first, then
   progress to 8×8 and 16×16.
4. **Run the dual-path comparison below.**

### Phase 4a: answer_vq_tokens vs answer_image — dual-path comparison

There are two ways to feed corrupted data into Lumina's MMU. Both should be
kept as first-class paths; the goal is to understand which one works better
and whether they complement each other.

**Path A — `answer_vq_tokens()` (discrete token input):**

```
clean VQ tokens → corrupt → corrupted VQ tokens
                                ↓
                  Lumina MMU (token space)
                                ↓
                  "which cells are corrupted?"
```

- Input: 64×64 grid of discrete token IDs (integers from VQ codebook)
- Pros: no decode/encode round-trip, lossless, preserves exact token identity,
  faster (no image generation step)
- Cons: model's MMU was pretrained on RGB images, not raw token grids;
  spatial relationships in token space may be harder to learn
- Current status: parse_rate=0.156 after LoRA (schema mismatch fix pending)

**Path B — `answer_image()` (decoded image input):**

```
clean VQ tokens → corrupt → corrupted VQ tokens
                                ↓
                          VQ-VAE decode
                                ↓
                          corrupted image (RGB)
                                ↓
                  Lumina MMU (image space)
                                ↓
                  "which region has artifacts?"
```

- Input: 1024×1024 RGB image (decoded from corrupted tokens)
- Pros: uses the model's native vision encoder (pretrained on natural images),
  spatial relationships are visually grounded, may leverage existing
  object/scene understanding
- Cons: adds VQ-VAE decode step (~1-2 sec per image), decode may smooth
  or hide subtle token-level corruptions
- Current status: **not yet tested on self-corruption data** (was verified
  working in June 2025 for general image description, not localization)

**Comparison experiment design (for Windows Codex):**

The SFT data pipeline should produce paired training examples for both paths
from the same corruption rows, so every LoRA experiment can be run on both
modalities and compared head-to-head.

| What to build | Purpose |
|---------------|---------|
| `--input-mode vq_tokens|decoded_image|both` flag on the SFT prep CLI | Generate training data for either or both paths from the same dataset |
| `--input-mode` flag on the localization probe CLI | Evaluate either path with the same metric suite |
| Per-sample `input_mode` field in probe_rows.jsonl | Track which path produced each prediction |
| Dual-path LoRA config | Train two LoRA adapters from the same SFT split, differing only in input modality |
| Comparison report CLI | Given two probe outputs (vq_tokens vs decoded_image), produce side-by-side metrics |

**Comparison metrics to report:**

| Metric | Why |
|--------|-----|
| parse_rate | Which path produces more parseable JSON |
| hit_any_rate, mean_iou | Which path localizes more accurately |
| mean_center_displacement | Which path gets closer to true corruption center |
| call_error_count | Which path has fewer model failures |
| inference_latency_ms per sample | Trade-off between speed and accuracy |
| GPU memory at inference | Both paths need to fit on one L40S |

**Decision rule after comparison:**

- If one path clearly dominates → make it the primary, keep the other as
  fallback/ablations
- If they complement (e.g., vq_tokens better at fine grids, image better at
  coarse grids) → design a fusion strategy
- If both are weak → revisit prompt engineering or training data before
  either path
- In all cases, **keep both code paths maintained** — the token-space path
  is unique to this project and may become important as Lumina's MMU evolves

**Server execution plan after Codex delivers:**

```bash
# Train on BOTH paths with matching SFT data
python -m ascr.cli.stage4_train_mmu_lora --config ...vq_tokens.yaml
python -m ascr.cli.stage4_train_mmu_lora --config ...decoded_image.yaml

# Evaluate both
python -m ascr.cli.stage4_mmu_localization_probe --config ...probe_vq_tokens.yaml
python -m ascr.cli.stage4_mmu_localization_probe --config ...probe_decoded_image.yaml

# Compare
python -m ascr.cli.stage4_compare_input_modes \
  --vq-tokens-probe .../probe_vq_tokens/summary.json \
  --decoded-image-probe .../probe_decoded_image/summary.json \
  --output-dir .../input_mode_comparison
```

### Suggested scripts for Windows Codex

> **Important: these are suggestions based on server-side observations.**
> The Windows Codex should do its own deep analysis of the codebase, the
> Phase 4 server results, and the dual-path trade-offs before deciding what
> to implement. The specs below are a starting point, not a requirement
> list. Skip, modify, or replace anything that doesn't fit the actual
> code structure.

---

#### Script 1: Fix SFT schema alignment (highest impact, lowest effort)

**Problem observed on server:** The LoRA outputs `{"correction_instruction":
"11,12,13,..."}` while the probe parser expects `{"corrupted_cells_16x16":
["J10","J9"]}`. The key name and value format don't match.

**Suggested fix:** Modify `ascr/cli/stage4_prepare_mmu_sft.py` or
`ascr/training/prepare_lumina_sft_data.py` so the generated SFT target JSON
uses the exact keys and cell-label format that the probe parser consumes:

```json
{
  "corrupted_cells_4x4": ["D3"],
  "corrupted_cells_8x8": ["H6", "H7"],
  "corrupted_cells_16x16": ["P12", "P13"]
}
```

The coarse labels already exist in the dataset (`coarse_labels_4x4`,
`coarse_labels_8x8`, `coarse_labels_16x16`) — they just need to be
propagated into the SFT target in the format the parser expects.

**Server validation after delivery:** Re-run 4b→4c→4d. Expect parse_rate
to jump from 0.156 to well above 0.5 if the schema fix is the main blocker.

---

#### Script 2: `--input-mode` on SFT prep and probe CLIs

**Existing CLIs to extend:**

`ascr/cli/stage4_prepare_mmu_sft.py` — add:
```
--input-mode {vq_tokens,decoded_image,both}  (default: vq_tokens)
```
- `vq_tokens`: current behavior, SFT examples use `vq_ids_path`
- `decoded_image`: SFT examples use `corrupted_image` (decoded RGB path)
- `both`: generates two sets of SFT examples in one pass

`ascr/cli/stage4_mmu_localization_probe.py` — already has `use_vq_tokens`
(bool). Add symmetric:
```
--input-mode {vq_tokens,decoded_image}  (default: vq_tokens)
```
- `vq_tokens`: calls `engine.answer_vq_tokens(corrupted_vq_ids, prompt)`
- `decoded_image`: calls `engine.answer_image(corrupted_image_path, prompt)`
- Records `input_mode` in each probe row and summary

---

#### Script 3: Dual-path LoRA training configs

Two new YAML files under `configs/stage4/self_corrupt/`:

`mmu_lora_train_hard64_vq_tokens.yaml`:
```yaml
# Same as mmu_lora_train_hard64.yaml but with:
input_mode: vq_tokens
# Uses token-space SFT data
```

`mmu_lora_train_hard64_decoded_image.yaml`:
```yaml
# Same structure but:
input_mode: decoded_image
# Uses image-space SFT data
# image_size likely needs to stay at 1024 for vision encoder
# May need bf16 to fit in 45GB
```

All other hyperparameters (epochs, lr, lora_r, lora_alpha, seed) identical
between the two configs so the comparison is controlled.

---

#### Script 4: Comparison CLI

`ascr/cli/stage4_compare_input_modes.py`:
```bash
python -m ascr.cli.stage4_compare_input_modes \
  --probes probe_vq_tokens/summary.json probe_decoded_image/summary.json \
  --labels "VQ Tokens" "Decoded Image" \
  --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64/input_mode_comparison
```

Outputs:
- `comparison.json` — side-by-side metrics for all evaluation dimensions
- `comparison.md` — human-readable table

Example output table:
```
Metric                  VQ Tokens   Decoded Image   Winner
parse_rate              0.156       0.XXX           ?
hit_any_rate (16x16)    0.0         X.XXX           ?
mean_iou                0.0         X.XXX           ?
call_error_count        0           X               ?
mean_latency_ms         1200        XXXX            ?
```

---

#### Script 5: Dual-path shell runner

`scripts/training/run_stage4_mmu_lora_dual.sh`:

```bash
# Runs both paths end-to-end:
# 1. SFT prep with --input-mode both
# 2. Lumina SFT convert for both
# 3. LoRA training for both (can be submitted as parallel sbatch jobs)
# 4. Evaluation for both
# 5. Comparison report
```

Supports env-var toggles so the server AI can skip steps that are already
done:
```bash
RUN_VQ_TOKENS=1 RUN_DECODED_IMAGE=1 bash scripts/training/run_stage4_mmu_lora_dual.sh
```

---

#### Script 6: Parallel dual-path Slurm wrapper

`jobs/stage4/train_mmu_lora_dual.sbatch`:

Heterogeneous job array — two job steps that can run in parallel on
different GPUs:
```bash
#SBATCH --job-name=ascr-s4-dual
# Two job steps, each 1 GPU
# Step 1: VQ tokens LoRA training
# Step 2: Decoded image LoRA training
# After both: comparison eval
```

Or simpler: two independent sbatch submissions from the shell runner,
each with its own job id. The comparison step waits for both.

---

#### Script 7: `answer_image()` smoke test on self-corruption data

Before full dual-path training, verify that `answer_image()` works on
corrupted images with a localization prompt:

`ascr/cli/stage4_image_mmu_smoke.py`:
```bash
python -m ascr.cli.stage4_image_mmu_smoke \
  --dataset outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl \
  --limit 8 \
  --output-dir outputs/stage4_self_corrupt/image_mmu_smoke
```

This is a lightweight check — does the model even understand the
question "where are the artifacts in this image?" when fed a decoded
corrupted image? If the answer is gibberish even with a simple prompt,
then the image path may need prompt engineering before LoRA training.

---

#### Script 8 (stretch): BF16 mixed-precision training support

To recover LoRA training at image_size=1024 on 45GB L40S:

In `ascr/training/stage4_mmu_lora.py`:
- Add `--torch-dtype bfloat16` option
- Add `--gradient-checkpointing` flag
- Add `--target-modules` CLI override (so server can tune without
  editing configs)

Expected memory savings: bf16 model weights ≈ 50% reduction.
With gradient checkpointing: another ~30% activation memory reduction.
These together should allow image_size=1024 with 4+ target modules.

---

#### Script 9 (stretch): Coarse-first curriculum configs

Three training stages, each building on the previous:

| Stage | Grid | Target cells | Expected difficulty |
|-------|------|-------------|-------------------|
| 1 | 4×4 | 1-2 cells | Easiest (16 cells total) |
| 2 | 8×8 | 1-4 cells | Medium (64 cells) |
| 3 | 16×16 | 1-8 cells | Hardest (256 cells) |

Configs: `mmu_lora_train_hard64_grid4.yaml`, `_grid8.yaml`, `_grid16.yaml`

Each stage trains a LoRA on the SFT data filtered to that grid's labels.
The server can then evaluate all three and see where accuracy degrades.

---

### Suggested priority order

```
1. Script 7 (image MMU smoke)        ← verify image path works at all
2. Script 1 (schema fix)             ← unblock parse_rate immediately
3. Script 2 (--input-mode flags)     ← enable controlled comparison
4. Script 3 + 5 + 6 (dual configs + runners) ← full comparison pipeline
5. Script 4 (comparison CLI)         ← quantify the difference
6. Script 8 (bf16)                   ← recover full-res training
7. Script 9 (curriculum)             ← improve localization accuracy
```

### What the server AI can do while waiting

Without waiting for any new code, the server AI can:
- Run the image MMU smoke test manually by passing `--prompt` and
  `--image` arguments if the probe CLI supports them
- Run selector baselines on any new dataset
- Submit parallel locality probe jobs at scale (manual sharding already
  proven to work)
- The Hard64 dataset (128 rows) and probe outputs are intact and
  verified

### Phase 4 Local Follow-up Implemented (2026-06-28)

Windows Codex integrated the server branch into `main` and implemented the
highest-priority follow-ups from the Phase-4 result:

- Stage-4 SFT now defaults to the compact `localization_cells` target schema:
  `has_error` plus `corrupted_cells_4x4`, `corrupted_cells_8x8`, and
  `corrupted_cells_16x16` when the primary grid is 16x16.
- `safe_parse_mmu_localization_payload()` normalizes `localization_cells`,
  legacy `SemanticEvaluation`, and the old bad `correction_instruction`
  integer-list pattern into ASCR `SemanticEvaluation` before scoring.
- `LuminaNativeEvaluator` can parse the compact localization schema and can be
  configured with `target_schema: localization_cells` for Stage-4 ASCR-loop
  smoke tests while Stage-2 remains on `semantic_evaluation` by default.
- SFT prep and probe CLIs now expose explicit
  `--input-mode {vq_tokens,decoded_image,both}` / `--target-schema` flags.
- `prepare_lumina_sft_data` now honors per-row `input_mode`; decoded-image SFT
  examples no longer silently fall back to direct VQ-token caches when
  `vq_ids_path` is also present.
- Added paired vq-token and decoded-image configs under
  `configs/stage4/self_corrupt/`, including L40S fallback configs that use
  `image_size=512`, `max_seq_len=2048`, and `q_proj,v_proj`.
- Added `ascr.cli.stage4_compare_input_modes` and
  `ascr.cli.stage4_image_mmu_smoke`.
- Added `scripts/training/run_stage4_mmu_lora_dual.sh` and
  `jobs/stage4/train_mmu_lora_dual.sbatch`.

The next server run should rerun SFT prep and LoRA training from scratch; do
not reuse the previous adapter because it was trained against the old target
schema.

### Phase 4.5: GC-first UMM/LoRA Scaleout

Server testing showed that 1024px LoRA training on a single L40S is activation
bound. 8-bit Adam reduces optimizer state but does not reduce the forward-pass
activation footprint enough. The current repo therefore treats gradient
checkpointing as a first-class Stage-4 training option rather than a manual
patch note.

Implementation:

- `ascr.training.train_lumina_lora_smoke` still tries native HuggingFace
  `model.gradient_checkpointing_enable()` when requested.
- If Lumina/LLaDA does not advertise support, `--gradient-checkpointing-fallback
  force` wraps detected decoder/block modules with `torch.utils.checkpoint`.
- Every LoRA `training_manifest.json` records `gradient_checkpointing_report`.
  A run where `wrapped_module_count` is 0 should be treated as not using real
  gradient checkpointing.

Server smoke:

```bash
EPOCHS=1 sbatch jobs/stage4/train_mmu_lora_gc_probe.sbatch
```

The Slurm job has only two array tasks to avoid the earlier
`QOSMaxSubmitJobPerUserLimit` issue:

| Task | Purpose |
|------|---------|
| 0 | hard64 vq-token 1024px full-module GC + adam8bit memory smoke |
| 1 | grid4 1024px full-module GC + adam8bit train/probe/failure analysis |

Manual commands:

```bash
python -m ascr.cli.stage4_train_mmu_lora \
  --config configs/stage4/self_corrupt/mmu_lora_train_hard64_vq_tokens_l40s_1024px_gc_adam8bit.yaml \
  --epochs 1

python -m ascr.cli.stage4_train_mmu_lora \
  --config configs/stage4/self_corrupt/mmu_lora_train_hard64_grid4_vq_tokens_l40s_1024px_gc_adam8bit.yaml \
  --epochs 1

python -m ascr.cli.stage4_mmu_localization_probe \
  --config configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid4_vq_tokens_l40s_1024px_gc.yaml
```

Probe failure analysis:

```bash
python -m ascr.cli.stage4_analyze_probe_failures \
  --probe-rows outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/probe_rows.jsonl \
  --summary outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/summary.json \
  --sft-examples outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/sft/train_sft_examples.jsonl \
  --train-jsonl outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/lumina_sft/train.jsonl \
  --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/failure_analysis
```

Registry:

```bash
python -m ascr.cli.stage4_build_run_registry \
  --roots outputs/stage4_self_corrupt \
  --output-dir outputs/stage4_self_corrupt/registry
```

One-command postprocess:

```bash
bash scripts/training/run_stage4_postprocess.sh
cat outputs/stage4_self_corrupt/next_actions/stage4_next_actions.md
```

The postprocess wrapper rebuilds the registry, analyzes the standard grid4 GC
probe when available, scans Stage-4 Slurm logs, and emits a concrete
`stage4_next_actions.md` decision file. This file is intentionally operational:
it includes shell commands for the next server-side action and the evidence
behind each branch.

Decision rules:

- If GC smoke fits and `gradient_checkpointing_report.backend` is
  `ascr_module_wrapper`, proceed to 15 epochs at 1024px.
- If GC smoke still OOMs but wrapped modules are nonzero, single-L40S is still
  too small; next architecture step is multi-GPU model/pipeline parallelism or
  shorter sequence length.
- If grid4 1024px trains to low loss but failure analysis remains dominated by
  `wrong_key_*` or `non_json_cell_label_text`, the bottleneck is format control
  rather than image resolution/capacity. Test stricter decoding or constrained
  JSON generation before scaling data.

Prompt/decoding sweep for format control:

```bash
sbatch jobs/stage4/stage4_probe_sweep.sbatch

# If QOS rejects 8 array tasks, split it:
sbatch --array=0-3 jobs/stage4/stage4_probe_sweep.sbatch
sbatch --array=4-7 jobs/stage4/stage4_probe_sweep.sbatch

MODE=summarize bash scripts/training/run_stage4_probe_sweep.sh
cat outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_sweep_l40s_1024px_gc/probe_sweep_summary.md
```

Default sweep:

| Axis | Values |
|------|--------|
| prompt_variant | `default`, `minimal_json`, `schema_first`, `schema_example` |
| max_new_tokens | `128`, `384` |
| answer_temperature | `0.0` |
| answer_cfg_scale | `0.0` |

This sweep is intentionally evaluation-only: it reuses the trained grid4 GC
adapter and tests whether the remaining failure is prompt/decoding format
control before launching more expensive data scaling or retraining.

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

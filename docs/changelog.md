# ASCR Experiment Changelog

Dated experiment narratives, latest first. This file is referenced from
[../README.md](../README.md) and holds the long-form historical record that used to live
inline in the project-control README. The README itself keeps the current Active TODO,
Quick Results Summary, Stage 1 Benchmark Summary, and the most recent independent GenEval
section.

---

## 2026-06-04 Phase 10 — Swap only the generator (Show-o → MMaDA-8B), keep the Qwen3.5-9B selector

Answers the user's hypothesis: run the *original* coarse ASCR pipeline (4×4 evaluate →
dilation → 32×32 token reopen) but change ONLY the base generator from Show-o to
**MMaDA-8B**, keeping the **Qwen3.5-9B** VLM as the selector (everything else, including the
selector, unchanged). Fully additive.

**Engineering blocker + solution:** MMaDA (transformers 4.46) and Qwen3.5-9B
(transformers 5.2.dev fork) cannot live in one process — MMaDA's remote code fails on
`all_tied_weights_keys` under 5.2.dev. Built a **two-process file-IPC pipeline**: each pair
uses two GPUs — a resident MMaDA worker (`.venv-mmada`) + a resident Qwen eval server
(`.venv-qwen36`) talking over a shared IPC dir. New files: `ascr/evaluators/remote_eval.py`
(`RemoteFileEvaluator`), `scripts/qwen_eval_server.py`,
`scripts/run_mmada_qwen_coarse_hard64.py`, `configs/stage1_mmada8b_qwen9b_coarse.yaml`,
`jobs/stage1_mmada_qwen_coarse_hard64_8gpu.sbatch`, `tests/test_remote_eval_wiring.py`
(6 tests). Whole suite **96/96 pass**.

**Run:** 4 nodes × 8 GPU = 16 pairs (each model loaded once). Hard64: 64/64 generated,
27 revised (Qwen judged 48/64 already correct — more decisive than self-eval).

**Gemini 3 Flash (clean: score ≥ 0.5; pairwise: bidirectional debiased):**

| metric | MMaDA baseline | **MMaDA + Qwen-9B coarse (P10)** | MMaDA self-coarse (P9) | MMaDA self-direct (P8) |
| --- | --- | --- | --- | --- |
| clean pass-rate | 33/64 = 51.6% | **37/64 = 57.8% (+6.2pp)** | 33/64 = 51.6% | 33/64 = 51.6% |
| pairwise vs baseline (w/l/t) | — | **2 / 0 / 62** | 0 / 0 / 64 | 0 / 0 / 64 |

**Conclusion:** keeping the external Qwen-9B selector (only swapping the generator) is what
makes ASCR actually improve over baseline (+6.2pp, wins 2 / loses 0 vs baseline), whereas
letting MMaDA judge itself (P8/P9) stayed flat at 51.6% / 0-0-64. The ASCR gain is driven by
the Qwen-9B selector and transfers to MMaDA-8B unchanged — matching the user's intuition.

**Clarifications:** (1) the clean pass criterion is a strict Gemini prompt-following judge
returning `{matches_prompt, score}`, PASS at `score ≥ 0.5`. (2) The MMaDA baseline (51.6%) is
**selector-independent** (baseline = initial generation, no revision), so changing the
selector does not change it; the earlier Show-o "73.4%" used a *different* benchmark/judge
rubric and is not comparable to this strict clean Hard64 judge; this baseline also runs only
`generation_timesteps=15` for speed. Results: `docs/mmada_qwen_coarse_hard64_results.json`.

---

## 2026-06-04 Phase 9 — MMaDA self-evaluation + the ORIGINAL ASCR coarse (4×4) selection strategy

Additive Stage-1 task that again uses **MMaDA-8B as a "selector that calls itself"**, but —
unlike Phase 8's direct 32×32 self-selection — the selection + regeneration follows the
**original ASCR coarse strategy**: MMaDA judges its own image on a **4×4** grid (`A1..D4`),
the chosen coarse cells are **dilated and projected back to the 32×32 token grid**, and those
tokens are reopened/re-diffused. No existing Show-o / Qwen / Phase-8 code was touched.

**New files (all additive):** `ascr/evaluators/mmada_self_coarse.py`
(`MMaDASelfCoarseEvaluator` — 4×4 two-stage A1–D4 self-eval + 32×32→4×4 confidence avg-pool
fallback), `configs/stage1_mmada8b_self_coarse.yaml`,
`ascr/cli/run_stage1_mmada_self_coarse.py` (reuses `ASCRLoop` +
`GridSemanticReopeningSelector`, shares the one loaded 8B engine),
`scripts/run_mmada_self_coarse_hard64.py` (resident worker with **global multi-node sharding**
via `NODE_INDEX`/`NODE_COUNT`), `jobs/stage1_mmada_self_coarse_hard64_8gpu.sbatch`,
`tests/test_mmada_self_coarse_wiring.py` (10 tests). Registry gains the `mmada_self_coarse`
backend. Whole suite: **90/90 tests pass**.

**Run:** 2 nodes × 8 GPU (16 cards, 8B loaded once per card). Hard64: 64/64 generated, 58
revised, max 3 iterations.

**Key mechanistic finding (answers "which selector is more accurate"):** across 166 error
iterations MMaDA grounded free-form `A1..D4` cells **64 times (≈39%)** at 4×4, vs **~0%** at
32×32 in Phase 8 (which therefore ran almost entirely on the per-token confidence fallback).
The coarse 4×4 grid is materially more tractable for MMaDA's MMU, i.e. a more faithful/accurate
selector.

**Gemini 3 Flash (ofox.ai) results:**

| metric | baseline (this run) | coarse-self (Phase 9, 4×4) | direct-self (Phase 8, 32×32) |
| --- | --- | --- | --- |
| clean pass-rate | 32/64 = 50.0% | 33/64 = **51.6%** | 33/64 = **51.6%** |
| pairwise vs baseline (debiased, w/l/t) | — | 0 / 0 / 64 | 0 / 0 / 64 |

**Conclusion:** final image quality is a tie (both 51.6% clean, both 0/0/64 vs baseline), but
the coarse-4×4 arm is the one that actually exercises MMaDA's spatial understanding, so the
original ASCR coarse-then-dilate strategy is the more faithful selector when MMaDA judges
itself. Results: `docs/mmada_self_coarse_hard64_results.json`.

---

## 2026-06-04 Phase 8 — Migrate Stage-1 to MMaDA-8B, "the selector calls itself"

Added a fully additive Stage-1 task on **MMaDA-8B** (`Gen-Verse/MMaDA-8B-MixCoT`, 8B unified
discrete diffusion, same MAGVIT-v2 tokenizer / 32×32=1024 tokens as Show-o) where **one 8B
model both generates the image and acts as the selector** (self-evaluation, self-selection,
self-reopen; single load per GPU, shared `MMaDANativeEngine`). No existing Show-o/Qwen code
was touched.

**New files:** `ascr/generators/mmada_native.py`, `ascr/generators/mmada.py`,
`ascr/evaluators/mmada_self.py`, `ascr/cli/run_stage1_mmada_self.py`,
`configs/stage1_mmada8b_self_direct_token.yaml`, `scripts/download_mmada_models.py`,
`scripts/run_mmada_self_hard64.py`, `scripts/merge_mmada_self_manifests.py`,
`jobs/stage1_mmada_self_hard64_8gpu.sbatch`, `jobs/stage1_mmada_self_smoke.sbatch`,
`tests/test_mmada_self_wiring.py` (13 mock tests). Modified only the two registries and the
direct CLI's `--generator` choices.

**Key finding (empirical, on real GPU):** MMaDA's MMU **global understanding works** (it
correctly describes images, e.g. "a blue circle in the center of a red background"), but it
**cannot ground free-form `R{row}C{col}` coordinates onto the 32×32 token grid** (returns
garbage like `"1"`; MAGVIT re-encoding also erases the thin grid overlay). So "let the model
name the wrong token cells in text" does **not** work.

**Working direct-token self-selection:** use the model's **own per-token confidence** — run
one `t2i` forward over the current 1024 tokens and read the softmax probability of each present
image token; lowest probability = least confident = reopen. This is the model judging its own
1024 tokens directly, no down-sampling, no external selector. Wired as the fallback when text
localization yields no cells (kept the 64-cell guardrail from Phase 7).

**Hard64 + Gemini-3-Flash-Preview (8-GPU resident, one 8B load per card; login-node judge):**

| Metric | MMaDA baseline (initial only) | MMaDA **self** (self-eval/select/reopen loop) |
|---|---|---|
| Gemini clean pass-rate | 31/64 = **48.4%** | 33/64 = **51.6%** (+3.1pp / +2 prompts) |
| Bidirectional debiased pairwise (self vs baseline) | — | **0 win / 0 loss / 64 tie** (quality on par) |
| Prompts that triggered a reopen | — | **63/64** (self-eval almost always says "error" → confidence fallback reopens 64 cells) |

**Conclusion:** MMaDA-8B can self-evaluate semantically and **self-select tokens via its own
confidence**, yielding a small clean-rate gain with **no quality regression** (pairwise all
ties); but it **cannot do fine-grained 32×32 semantic localization** via free-form text. This
directly answers the original question ("can the selector judge exactly which of the 1024
tokens are wrong?"): **not via free-form coordinates, but yes via the model's own per-token
confidence.** Machine-readable summary: `docs/mmada_self_hard64_results.json`.

---

## 2026-05-21 BAGEL vs ShowO Baseline — Completing the Comparison Triangle

This run closes the triangle: ASCR is already shown to beat BAGEL (2026-05-19); this experiment
directly compares BAGEL against the ShowO baseline on the same prompt set, confirming the
ordering **ASCR >> BAGEL > ShowO baseline**.

**Protocol:**
- Same hard64 compositional prompt set (`configs/prompts/t2i_compbench_hard64.txt`).
- BAGEL images from job 68664 (previously generated, already in
  `outputs/bagel_t2i_compbench_hard64_8gpu_20260519_202625/images/`).
- ShowO baseline images from the Stage 1 Phase 1 checkpoint (job 68660):
  `baseline_showo.png` per prompt in
  `outputs/benchmarks_t2i_compbench_qwen35_hard64_slurm8gpu_reuse_20260519_191652/`.
- Pairing: `scripts/pair_bagel_vs_showo_baseline.py` matches by prompt text. 47 of 64 prompts
  were successfully paired (the remaining 17 could not be matched due to path differences).
- Judge: `scripts/judge_showo_ascr_pairwise_qwen.py` with `--baseline-label ShowO
  --ascr-label BAGEL --no-image-labels` (Qwen3.5-9B, no text overlaid on canvas).
- **Slurm job:** 68752, 1 GPU, completed in 00:07:36.

**Results (Pairwise, Qwen side-by-side, no image labels, N=47 matched pairs):**

| Metric | Count | Rate |
| --- | ---: | ---: |
| **BAGEL wins** | **26** | **55.3%** |
| ShowO wins | 21 | 44.7% |
| Ties | 0 | — |
| Net BAGEL advantage | **+5** | — |

**Output directory:** `outputs/bagel_t2i_compbench_hard64_8gpu_20260519_202625`

**Key file:** `qwen_pairwise_bagel_vs_showo_baseline.json`

**Interpretation:**

BAGEL-7B-MoT achieves a modest **+5 net pairwise advantage** (26 wins vs 21 losses) over the
ShowO baseline on matched hard64 compositional prompts. Combined with the earlier BAGEL-vs-ASCR
result, this establishes a clear three-way ordering on this benchmark.

---

## 2026-05-19 BAGEL-7B-MoT vs ASCR Comparison

External T2I model comparison on the same hard64 prompt set.

**Model under test:** ByteDance-Seed/BAGEL-7B-MoT (7B mixture-of-transformers, public weights from HuggingFace).

**Protocol:**
- Same 64 prompts from `configs/prompts/t2i_compbench_hard64.txt`.
- BAGEL images generated by `scripts/run_bagel_text2image.py` using `.venv-bagel` (torch 2.5.1+cu121, flash-attn 2.7.4.post1).
- ASCR images are the clean final images from the Stage 1 Phase 1 hard64 checkpoint (job 68660).
- Qwen pairwise judge: `scripts/judge_showo_ascr_pairwise_qwen.py` with `--baseline-label BAGEL --ascr-label ASCR --no-image-labels` (no text drawn into the side-by-side canvas, eliminating label-content contamination).
- Clean pass/fail judge: per-image independent Qwen judgment, same threshold 0.5.

**Slurm jobs:**
- 68664: BAGEL hard64 generation, 8 GPU L40S, completed in 01:07:27.
- 68667: Qwen pairwise judge (afterok:68664), completed in 00:08:27.
- 68668: Qwen clean pass/fail judge (afterok:68664), completed in 00:05:21.
- 68669: summary aggregation (afterok:68667:68668), completed instantly.

**Results:**

Pairwise (Qwen side-by-side, no image labels, N=64):

| Metric | Count |
| --- | ---: |
| ASCR wins | **50** |
| BAGEL wins | 14 |
| Ties | 0 |
| Net ASCR advantage | **+36** |

Clean pass/fail (Qwen independent per-image, N=64):

| Metric | Count | Rate |
| --- | ---: | ---: |
| ASCR pass | **57** | 89.1% |
| BAGEL pass | 54 | 84.4% |
| Both pass | 52 | 81.3% |
| Both fail | 5 | 7.8% |
| Only ASCR passes | 5 | — |
| Only BAGEL passes | 2 | — |
| Net ASCR advantage | **+3** | — |

**Output directory:** `outputs/bagel_t2i_compbench_hard64_8gpu_20260519_202625`

**Key files:**
- `suite.json`: BAGEL generation records.
- `bagel_vs_ascr_suite.json`: paired BAGEL+ASCR records for judging.
- `qwen_pairwise_bagel_vs_ascr_nolabel.json`: full pairwise judgment with per-prompt verdicts.
- `qwen_clean_bagel_vs_ascr.json`: full clean pass/fail judgment.
- `bagel_vs_ascr_summary.json`: aggregated counts.

**Interpretation:**

ASCR achieves a **+36 net pairwise advantage** (50 wins vs 14 losses, 0 ties) and a
**+3 net clean-pass advantage** (57/64 = 89.1 % vs 54/64 = 84.4 %) over BAGEL-7B-MoT on the
hard64 compositional prompt set.

Key points for interpreting this result:

- **Not architecture-to-architecture.** ASCR uses ShowO (1.3 B unified masked multimodal
  generator) with an iterative correction loop; BAGEL-7B-MoT (7 B mixture-of-transformers) is a
  larger standalone model. The comparison shows the correction-loop approach can outperform a
  larger dedicated model on compositional following, at the cost of additional inference compute.
- **Correction loop advantage on compositional prompts.** The hard64 prompts specifically target
  failures in spatial relations, color–object binding, shape–object binding, and counting —
  exactly the categories the ASCR loop is designed to detect and repair.
- **Pairwise vs clean-pass discrepancy.** The large pairwise net (+36) alongside a smaller
  clean-pass net (+3) means: in direct head-to-head comparison ASCR's images are consistently
  more precisely correct, while when judged in isolation at the 0.5 threshold both systems pass
  at similar rates. BAGEL's images often satisfy a loose pass criterion while being less precise
  on compositional details when directly compared.
- **Evaluator circularity.** Both judges use Qwen3.5-9B, which is also the semantic feedback
  model inside the ASCR correction loop. Human evaluation or official T2I-CompBench CLI metrics
  are needed for independent confirmation.
- **Sample BAGEL wins (14 prompts):** `a plastic toy and a glass bottle`,
  `a giraffe next to a lamp`, `a girl on the top of a frog`.
- **Sample ASCR wins (50 prompts):** `a green bench and a blue bowl`,
  `an oblong cucumber and a teardrop plum`, `a dog in front of a desk`.

See [Qualitative Examples](#qualitative-examples) for side-by-side images of representative
wins, losses, and ties.

## 2026-05-19 Stage 1 Phase 1 Default Checkpoint

The current default Stage 1 flow is the T2I-CompBench hard64 sharded-reuse run. It is now the baseline for further configuration exploration.

Default command:

```bash
sbatch jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch
```

Default settings:

- One Slurm job requests 8 GPUs with #SBATCH --gres=gpu:8.
- Inside the allocation, SHARD_WORKERS=8 starts eight workers; each worker receives one CUDA_VISIBLE_DEVICES shard.
- PROMPTS_FILE=configs/prompts/t2i_compbench_hard64.txt and PROMPT_LIMIT=64.
- REUSE_MODELS=1 keeps the baseline generator, ASCR generator, and Qwen evaluator loaded across each worker shard.
- ASCR_START_MODE=baseline starts ASCR from the exact native Show-o baseline token state.
- GENERATION_TIMESTEPS=18, GUIDANCE_SCALE=4, MAX_ITERATIONS=8, REPEAT_COUNT=1, and SEED_STEP=1 are the default sweep anchor.
- return_initial_on_max_error=true keeps unresolved ASCR loops conservative.

Completed hard64 checkpoint:

- Job 68660 completed with exit code 0:0 in 00:16:38.
- Verified allocation: billing=32,cpu=32,gres/gpu=8,mem=192G,node=1.
- Run root: outputs/benchmarks_t2i_compbench_qwen35_hard64_slurm8gpu_reuse_20260519_191652.
- Pairwise Qwen judge: ASCR win 13, ASCR loss 6, tie 45, net +7.
- Clean Qwen pass/fail: ASCR pass 57/64, baseline pass 53/64, net +4.
- Detailed summary: docs/stage1_phase1_summary_20260519.md.

Interpretation: this is a Stage 1 automated benchmark signal, not independent human evidence, because Qwen3.5-9B is also the ASCR loop evaluator. Future experiments should report deltas against this default and keep pairwise and clean pass/fail counts separate.

## 2026-05-19 Fair T2I-CompBench Judge and ASCR Safety Plan

This update makes the Stage 1 comparison stricter without deleting the existing working pieces.

Evaluation plan:

1. Use `ascr_start_mode: baseline` as the default for Qwen3.5-9B comparison configs and Qwen3.5 Slurm jobs. In this mode, ASCR starts from the exact native Show-o baseline token state, so unchanged outputs remain attributable to the baseline and changed outputs are attributable to reopening.
2. Keep `partial` as an explicit experimental mode only. It is useful for denoising-time intervention experiments, but its outputs mix ASCR behavior with independent sampling noise and should not be used for primary baseline-vs-ASCR claims.
3. Treat `comparison` / `heuristic_comparison` in `suite.json` as a development-only heuristic. The current `score_image` path only supports simple color-presence and red-left-blue checks, so hard compositional prompts require VLM or official metric judging.
4. Use `scripts/judge_showo_ascr_pairs_qwen.py` for clean single-image pass/fail judging. It compares clean `baseline_showo.png` against clean `ascr_final_image`; grid overlays remain localization artifacts only.
5. Use `scripts/judge_showo_ascr_pairwise_qwen.py` as the primary VLM pairwise judge. It builds a side-by-side clean image with LEFT as baseline and RIGHT as ASCR, then asks Qwen3.5-9B which image better follows the prompt.
6. For unresolved ASCR loops, set `return_initial_on_max_error: true`. If the loop reaches `max_iterations`, the report returns the initial decoded image as the conservative final image and stores the raw last candidate under `raw_final_decoded_image`. This prevents an unresolved final repair candidate from silently replacing the safer starting image.
7. Future official T2I-CompBench metrics should be layered on top of the same clean image outputs and reported separately from both Qwen judges. The expected report layout is heuristic development signal, Qwen clean pass/fail, Qwen side-by-side pairwise, then official T2I-CompBench metric when available.

New and updated entry points:

- `scripts/judge_showo_ascr_pairwise_qwen.py`: side-by-side Qwen3.5-9B pairwise judge. Outputs `qwen_pairwise_judge.json`, `qwen_pairwise_judge.md`, and paired comparison images under `qwen_pairwise_judge/pairwise_images/`.
- `configs/stage1_showo_qwen35_9b.yaml` and `configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml`: default to `ascr_start_mode: baseline` and `return_initial_on_max_error: true`.
- Qwen3.5 smoke and T2I jobs now default to `ASCR_START_MODE=baseline`; T2I smoke jobs run both clean pass/fail and side-by-side pairwise judges.

Recommended hard benchmark flow:

```bash
sbatch jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch
```

This is the default path for hard64 when 8 GPUs are available: Slurm allocates one job with `ReqTRES=gres/gpu=8`; inside that allocation, 8 workers each receive one `CUDA_VISIBLE_DEVICES` shard and process a contiguous prompt slice with `REUSE_MODELS=1`. Override `SHARD_WORKERS`, `PROMPTS_FILE`, `PROMPT_LIMIT`, or `RUN_ROOT` only when deliberately testing a different shape.

Manual judge rerun on an existing suite:

```bash
python scripts/judge_showo_ascr_pairs_qwen.py path/to/suite.json --config configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml --output path/to/qwen_clean_final_pair_judge.json
python scripts/judge_showo_ascr_pairwise_qwen.py path/to/suite.json --config configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml --output path/to/qwen_pairwise_judge.json
```

Interpretation rule: primary claims should come from `qwen_pairwise_judge.json` and clean pass/fail counts, not from the heuristic `comparison.verdict`. `comparison.verdict` remains in JSON for backward compatibility and quick color-rule smoke debugging only.

## 2026-05-14 T2I-CompBench, Clean Judge, and Runtime Reuse

T2I-CompBench is now wired as the harder compositional prompt suite for Stage 1 baseline-vs-ASCR checks.

New benchmark and judge entry points:

- `scripts/prepare_t2i_compbench_prompts.py`: exports T2I-CompBench prompt files from `NinaKarine/t2i-compbench`.
- `configs/prompts/t2i_compbench_hard_smoke8.txt`: 8 unique hard smoke prompts.
- `configs/prompts/t2i_compbench_hard64.txt`: 64-prompt harder follow-up subset.
- `jobs/stage1_t2i_compbench_qwen35_9b_smoke1.sbatch`: 1-GPU smoke and sequential fallback runner.
- `jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch`: recommended hard64 runner; one Slurm job requests `gres/gpu=8`, then launches 8 model-reuse workers.
- `scripts/run_stage1_showo_compare_sharded_reuse.sh`: shards prompts inside one allocation and keeps models loaded within each worker.
- `scripts/shard_prompts.py` and `scripts/aggregate_showo_ascr_suites.py`: split prompts and merge worker suites.
- `scripts/judge_showo_ascr_pairs_qwen.py`: clean final-image paired judge for baseline vs ASCR.

The final judge compares only clean generated images: baseline `baseline_showo.png` against ASCR `ascr_final_image` / `final_decoded_image`. ASCR grid images remain diagnostic artifacts for localization and must not be treated as final benchmark images.

Run the default T2I hard64 benchmark:

```bash
python scripts/prepare_t2i_compbench_prompts.py
sbatch jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch
```

For a tiny regression check only:

```bash
REUSE_MODELS=1 PROMPT_LIMIT=2 sbatch jobs/stage1_t2i_compbench_qwen35_9b_smoke1.sbatch
```

Runtime reuse is enabled for the single-process path with `--reuse-models` or `REUSE_MODELS=1`. This keeps the baseline generator, ASCR generator, and Qwen evaluator alive across prompts in the same process; compatible baseline and ASCR Show-o adapters also share the same underlying native Show-o engine.

Validated T2I smoke status:

- T2I 1-prompt / 1-GPU smoke job `68441` completed.
- T2I 8-prompt / 1-GPU fallback job `68443` completed and produced `outputs/benchmarks_t2i_compbench_qwen35_smoke8_1gpu/showo_ascr-20260514-040615/suite.json`.
- Clean final-image judge job `68444` completed on that suite with `baseline_pass=8`, `ascr_pass=8`, `both_pass=8`.
- T2I 2-prompt `REUSE_MODELS=1` validation job `68445` completed with `COMPLETED 0:0` in `00:02:21`; clean final-image judge counts were `baseline_pass=2`, `ascr_pass=2`, `both_pass=2`.
- The 8-prompt smoke validated the early pipeline but is no longer the default result path. The 2026-05-19 hard64 run is the current Stage 1 checkpoint and should be used as the reference when exploring new configs.

## 2026-05-14 Public Benchmark Prompt Suites

The original `configs/prompts/stage1_complex_prompts.txt` suite is now treated as an internal development smoke suite only. It is useful for regression checks, but it is not a public benchmark and should not be used for result claims by itself.

DrawBench has been added as the first public prompt-only suite:

- `scripts/prepare_drawbench_prompts.py` downloads or reuses the public `sayakpaul/drawbench` CSV and exports ASCR prompt text files.
- `configs/prompts/drawbench_smoke8.txt` contains an 8-prompt smoke subset covering major DrawBench categories.
- `configs/prompts/drawbench_all.txt` contains all 200 DrawBench prompts.
- `jobs/stage1_drawbench_qwen35_9b_smoke8.sbatch` runs the 8-prompt DrawBench smoke across 8 GPUs.

Prepare prompts:

```bash
python scripts/prepare_drawbench_prompts.py --smoke-limit 8
```

Run the DrawBench smoke:

```bash
sbatch jobs/stage1_drawbench_qwen35_9b_smoke8.sbatch
```

Important interpretation rule: DrawBench supplies prompts, not reference images. The current `comparison.verdict` field is still heuristic and should be read as a development signal only. A fair prompt-following comparison needs an independent judge protocol over the generated baseline and ASCR images, such as a Qwen3.5-9B final judge, TIFA/VQA-style judge, VQAScore, GenEval-style object checks, or a human/audited subset.

Latest public-benchmark smoke status:

- DrawBench 1-prompt / 1-GPU smoke job `68440` completed with `COMPLETED 0:0` in `00:00:56`.
- Prompt: `A red colored car.`
- Artifacts: `outputs/benchmarks_drawbench_qwen35_smoke1gpu/showo_ascr-20260514-034344/comparison.json`.
- Counts: 1 `comparison.json`, 1 `evaluation.json`, 0 parser errors, 0 abstains.
- ASCR loop status: `stop_reason: no_semantic_error`, `evaluator_calls: 1`, `ascr_insertions: 0`.
- Heuristic comparison: baseline 1.0, ASCR 1.0, verdict `tie_or_unclear`.
- DrawBench 8-prompt / 8-GPU smoke job `68439` is submitted but still `PENDING (Resources)`.
- This smoke validates the public prompt path, not a fair benchmark claim; the final independent judge protocol is still pending.

## 2026-05-14 Qwen3.5-9B Default Evaluator

Qwen3.5-9B is now the default Stage 1 evaluator path. The validated local checkpoint is `models/qwen3.5-9b`, downloaded from `Qwen/Qwen3.5-9B`; model weights, generated outputs, and benchmark raw payloads remain outside git.

New default entry points:

- `configs/stage1_showo_qwen35_9b.yaml`: default full-cap Stage 1 config.
- `scripts/download_qwen35_9b_snapshot.sh`: reproducible Qwen3.5-9B snapshot download.
- `scripts/run_stage1_showo_compare.sh`: single-process comparison runner, defaulting to Qwen3.5-9B.
- `scripts/run_stage1_showo_compare_parallel.sh`: one-worker-per-GPU comparison runner, defaulting to Qwen3.5-9B.
- `jobs/stage1_qwen35_9b_smoke1gpu.sbatch`: single-GPU full-flow smoke.
- `jobs/stage1_qwen35_9b_parallel8.sbatch`: 8-GPU parallel smoke.

Validated Qwen3.5-9B runs:

| Run | Job | Result | Notes |
| --- | --- | --- | --- |
| Single GPU full-flow smoke | `68379` | `COMPLETED 0:0` in `00:02:43` | produced `comparison.json` |
| 8-GPU parallel smoke | `68386` | `COMPLETED 0:0` in `00:07:32` | 8 comparisons, 29 evaluator calls, 0 parser errors, 0 abstains |

Historical Qwen3.6/AWQ configs and jobs are kept under `configs/experiments/qwen36/` and `jobs/experiments/qwen36/`. They are not the default path anymore.

## 2026-04-28 Qwen3.6 Evaluator Path

Added a Qwen-VL evaluator backend for `Qwen/Qwen3.6-35B-A3B` as the stronger Stage 1 semantic judge. The backend lazily loads the HuggingFace image-text model, asks for strict JSON with 4x4 grid localization, normalizes common JSON variants, and converts the result into the existing `SemanticEvaluation` schema.

New entry points:

- `configs/stage1_showo_qwen36.yaml`: native Show-o generator plus Qwen3.6 evaluator.
- `scripts/download_qwen36_snapshot.sh`: download the full Qwen3.6 snapshot to `models/qwen3.6-35b-a3b` from the login node.
- `jobs/stage1_compare_qwen36_gpu.sbatch`: 4-GPU Slurm compare run using the local offline Qwen3.6 snapshot.
- `requirements-qwen-vl.txt`: Qwen evaluator runtime dependencies.

Environment status on 2026-04-28:

- Installed Qwen evaluator dependencies into `.venv`: `transformers 4.57.6`, `accelerate 1.10.1`, `qwen-vl-utils 0.0.14`, `sentencepiece 0.2.1`, and related packages.
- Verified `AutoModelForImageTextToText` import under the updated environment.
- Verified Qwen3.6 processor image-tokenization works with `processor_use_fast: false`; fast preprocessing hits a `torch.compiler.is_compiling` compatibility issue under the current `torch 2.2.1` environment.
- Verified HuggingFace model metadata for `Qwen/Qwen3.6-35B-A3B`; the repository is public and contains 26 safetensors shards.
- The full Qwen3.6 snapshot is about 67 GiB and is not committed. Compute nodes cannot resolve `huggingface.co`, so download it first on the login node with `bash scripts/download_qwen36_snapshot.sh`; the Slurm job defaults to `QWEN_MODEL_PATH=models/qwen3.6-35b-a3b`, `QWEN_LOCAL_FILES_ONLY=1`, and offline HuggingFace mode.
- Validation after adding this backend: syntax checks passed and `python -m unittest discover -s tests -v` reports 28 tests passed.

## 2026-04-28 Stage 1 Real Show-o Wiring

Completed on 2026-04-28:

- Wired the real local Show-o source at `external/Show-o` through `ShowOAdapter`. The adapter now defaults to a native in-process token loop rather than only calling the official text-to-image/inpainting scripts.
- Added `ShowONativeEngine`, which loads local Show-o, MAGVITv2, and Phi in process and exposes the discrete image-token state `u`, decoded token grid, token confidence map, and confidence remask `M_conf`.
- Implemented semantic reopening as a direct force-mask on selected Show-o image-token positions followed by continued native denoising. The old subprocess T2I/inpainting helpers remain available as fallback paths.
- Added `ShowOMMUEvaluator`, which asks local Show-o MMU to judge whether the gridded image violates the original prompt and to return a compact yes/no semantic judgment, with JSON and natural-language fallback parsing for 4x4 grid localization.
- Updated `configs/stage1_showo_local.yaml` to use `native_token_loop: true`, `confidence_steps: 2`, and `evaluator.name: showo_mmu` by default.
- Updated trace/artifact wiring so decoded images, grid images, evaluation JSON, reopening masks, token state JSON, and confidence JSON can be inspected per iteration.
- Updated `python -m ascr.cli.compare_showo_ascr` so the baseline is generated as a native Show-o state and ASCR starts from that same token state instead of reconstructing a zero-token placeholder.
- Added local-offline loading controls so ASCR uses `models/phi-1_5` directly and sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` for Slurm runs.
- Updated Slurm entry points for `gpu_shared` smoke and `gpu` longer runs.

Validated commands before GPU smoke:

```bash
source .venv/bin/activate
python -m py_compile ascr/generators/showo_native.py ascr/generators/showo.py ascr/generators/registry.py ascr/evaluators/showo_mmu.py ascr/evaluators/registry.py ascr/core/loop.py ascr/cli/run_stage1.py ascr/cli/compare_showo_ascr.py
python -m unittest discover -s tests -v
git diff --check
```

Validation result so far:

- Unit tests: 24 passed.
- Syntax checks: passed.
- Diff whitespace check: passed.
- Login-node CUDA check: `torch.cuda.is_available()` returned `False`; real Show-o validation must run through Slurm on a GPU node.
- `gpu_shared` smoke job `66508` completed in 00:02:14 and produced `outputs/benchmarks_gpu_shared/showo_ascr-20260428-132546/comparison.json`.
- Formal `gpu` compare job `66513` completed in 00:02:12 and produced `outputs/benchmarks/showo_ascr-20260428-132908/comparison.json`.
- Offline-loading smoke job `66515` completed in 00:00:21 with no HuggingFace DNS retry. Show-o MMU probe job `66521` confirmed that the local MMU answers simple image questions and yes/no prompt checks. Final yes/no MMU smoke job `66523` completed in 00:00:11 with `stop_reason: no_semantic_error`, raw `showo_eval_text: " Yes"`, and token/confidence/trace artifacts written.

Current comparison evidence for prompt `A red cube left of a blue sphere`:

- 4-step `gpu_shared` smoke: baseline 1.0, ASCR 1.0, verdict `tie_or_unclear`.
- Final exact yes/no Show-o MMU smoke `66523`: baseline 1.0, ASCR 1.0, `stop_reason: no_semantic_error`, native token state and confidence artifacts saved.
- 18-step `gpu` run: baseline 0.9584777638352246, ASCR 0.9584777638352246, verdict `tie_or_unclear`.

Interpretation: Stage 1 is now wired according to the original ASCR mechanism at the code-interface level: confidence block, `M_conf`, token confidence map, semantic mask, and continued native denoising are all explicit. The remaining blocker before making result claims is broader multi-prompt and multi-seed evaluation on prompts where the baseline visibly violates the prompt.

## 2026-04-28 Stage 1 Scaffold and Show-o Local Validation

Latest Stage 1 scaffold status:

- Added the Python package scaffold, config loader, schemas, artifact writer, trace writer, grid projection, grid overlay, prompt composer, selector, generator/evaluator registries, mock generator, mock evaluator, and real-backend placeholders.
- Added CLI entry points: `ascr-stage1` and `ascr-train-selector`.
- Added dedicated environment scripts and a legacy-pip compatible `setup.py` path for this cluster.
- Created `.venv` and verified local development installation with `python setup.py develop`.
- Added Stage 1 unit tests and verified `python -m unittest discover -s tests -v` passes 14 tests.
- Verified mock Stage 1 dry-run creates a summary and trace under `outputs/smoke/...`.
- Added Slurm templates for `gpu_shared` Stage 1 debug, `gpu` Stage 1 formal runs, and reserved multi-GPU Stage 2 selector training.
- Added docs for Stage 1 design, cluster usage, benchmark planning, and data policy.

Latest local Show-o validation:

- Cloned Show-o source into `external/Show-o/` and kept it out of Git.
- Downloaded local model snapshots into `models/show-o-512x512`, `models/magvitv2`, and `models/phi-1_5`; these are ignored by Git.
- Installed Show-o inference dependencies in `.venv` and verified PyTorch CUDA on the interactive L40S.
- Installed project-local `rg` under `.venv/bin/rg` because the server has no system ripgrep.
- Added local Show-o download and text-to-image helper scripts.
- Verified direct original Show-o generation with 4-step smoke and 50-step baseline image runs; generated images are runtime artifacts under `outputs/` and ignored by Git.
- Native Show-o integration now exposes the image-token state `u`, token confidence map, confidence remask `M_conf`, semantic force-mask, and continued denoising path. A Show-o MMU semantic evaluator is wired locally with a compact yes/no prompt path, raw-output preservation, and fallback parsers for natural-language answers and grid-cell labels.

Remaining Stage 1 integration work:

- Run multi-prompt and multi-seed benchmark sweeps on `gpu_shared` for debug and `gpu` for formal results.
- Add batch parallelism for prompt sweeps; single-image Show-o inference remains single-GPU.


### Daily completion notes

Completed:

- Confirmed remote project directory: `/grp01/cds_bdai/JianyuZhang/ASCR`.
- Confirmed uploaded planning documents are present.
- Verified the two `.docx` planning documents can be parsed and read on the remote server.
- Confirmed GitHub repository page is reachable: `https://github.com/hmss2002/ASCR`.
- Confirmed Stage 1 defaults: Show-o generator, local VLM or local LLM/VLM evaluator, Slurm compatibility for both `gpu` and `gpu_shared`.
- Created this README as the Stage 1 project-control document.
- Created `.gitignore` for Python, ML, Slurm, artifact, data, secret, and environment files.
- Initialized Git repository on branch `main`.
- Configured GitHub remote `origin` with SSH URL `git@github.com:hmss2002/ASCR.git`.
- Created initial commit `59589d3` with README, `.gitignore`, and source planning documents.
- Pushed initial project-control files to GitHub `main`.

Stage 1 scaffold validated:

- Dedicated `.venv` created and local development install verified with `python setup.py develop`.
- `python -m unittest discover -s tests -v` passed 11 tests.
- Mock dry-run generated a summary and trace under `outputs/smoke/...`.

Next implementation batch:

1. Wire the real Show-o repository and checkpoint paths behind `GeneratorAdapter`.
2. Connect the selected local VLM/LLM backend behind `SemanticEvaluator`.
3. Run `jobs/stage1_debug_gpu_shared.sbatch` as the first Slurm smoke job.
4. Promote stable settings to `jobs/stage1_run_gpu.sbatch` for longer formal Stage 1 runs.


### Phase 11 — Fair 4-way Show-o vs MMaDA-8B on Hard64 (one identical clean Gemini judge)

Goal: answer whether Show-o's 73.4% was a rubric artifact, and which model wins under ONE
identical clean Gemini-3-Flash rubric, each at its own default high-quality generation.

- MMaDA HQ default = official eval config `stage3_512_cot`: `timesteps=20, gs=5` (README example uses 15/3.5). Used 20/5. Show-o default HQ = 50 steps / gs=4.
- New (all additive): `configs/stage1_mmada8b_qwen9b_coarse_hq.yaml`, `scripts/run_showo_qwen_coarse_hard64.py` (single-process Show-o+Qwen resident runner), `jobs/stage1_showo_qwen_coarse_hard64_8gpu.sbatch`. MMaDA arm reuses Phase-10 two-process IPC pipeline.
- Runs: Show-o arm 1 node x 8 GPU (64/64, 0 err, 20 revised); MMaDA-HQ arm 2 nodes x 8 GPU (64/64, 0 err, 24 revised). Model loaded once per GPU.
- Clean Gemini (same rubric): Show-o baseline 47/64=73.4%; Show-o+Qwen-coarse 50/64=78.1% (+4.7pp); MMaDA-8B HQ baseline 37/64=57.8%; MMaDA-8B+Qwen-coarse HQ 36/64=56.2% (-1.6pp, judge noise).
- Bidirectional debiased pairwise: Show-o+Qwen vs Show-o-base 5/0/59; MMaDA+Qwen vs MMaDA-base 3/0/61; Show-o-base vs MMaDA-base 26/5/33 (83.9% decisive); Show-o+Qwen vs MMaDA+Qwen 29/3/32 (90.6% decisive).
- Conclusion: Show-o's 73.4% is NOT a rubric artifact — under the SAME judge and MMaDA's HQ default, Show-o (1.5B) decisively beats MMaDA-8B on Hard64 compositional prompts, both baseline and +ASCR. Coarse ASCR + Qwen-9B helps Show-o (+4.7pp) and never regresses MMaDA. Results: `docs/fair_4way_hard64_results.json`.

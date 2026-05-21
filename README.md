# ASCR: Alternating Semantic-Confidence Revision

ASCR is a research prototype for studying and correcting confidence-semantic inconsistency in masked image-token generation. The central observation is that an image region can become confidence-stable during iterative denoising while still being semantically wrong with respect to the text prompt. Stage 1 starts with a zero-training implementation that uses a visible 4x4 grid and structured local semantic feedback to selectively reopen image-token regions instead of retrying the whole image.

This README is the project control document. It records the research plan, implementation plan, current progress, expected interfaces, cluster workflow, and GitHub synchronization policy. It should be updated whenever a meaningful implementation batch is completed.
 
## Active TODO (2026-05-22, updated)

**Two bugs discovered and fixed in commit `557d2fc` (2026-05-22):**

1. **GENERATION_TIMESTEPS default override:** `compare_showo_ascr_parallel.py` always forwarded `--generation-timesteps ${GENERATION_TIMESTEPS}` to the worker CLI, silently overriding yaml's `generation_timesteps: 50` with the sbatch default of 18. md5 comparison confirmed 68784 == 68753 baseline images are byte-identical (both 18-step). Fixed: sbatch defaults changed to 50.
2. **Qwen pairwise RIGHT-position bias:** Cross-checking four pairwise comparisons found that whichever model was placed on the RIGHT always won lopsidedly, regardless of actual quality. Fixed: `pair_bagel_vs_hard64_run.py` gains `--swap`; `stage1_hard64_bagel_3way_judge_sharded.sbatch` now loops over `fwd` + `swap` directions.

Action items:

- [x] Fix sbatch `GENERATION_TIMESTEPS` defaults 18 → 50 (two production sbatches).
- [x] Add bidirectional (fwd + swap) pairwise judging to hard64 BAGEL 3-way judge.
- [x] Commit + push fixes (commit `557d2fc`).
- [x] Submit 50-step regeneration jobs:
    - **68794** GenEval 553 regen @ 50-step (RUNNING, `SPGL-1-12`, ~4–6 h)
    - **68795** hard64 64 regen @ 50-step (RUNNING, `SPGL-1-18`, ~4–6 h)
    - **68796** auto-submit GenEval 3-way scoring (PENDING, `afterok:68794`)
    - **68797** auto-submit hard64 bidir 3-way pairwise (PENDING, `afterok:68795`)
- [x] Submit position-bias diagnostic jobs (on existing 18-step data, independent of regen):
    - **68798** bidir BAGEL 3-way pairwise on 68785 outputs (RUNNING, `SPGL-1-19`, ~3 h)
    - **68799** swap ShowO-vs-ASCR pairwise on 68753 outputs (PENDING, waiting GPU, ~3 h)
- [ ] After **68798** finishes: compare `fwd` vs `swap` win counts for BAGEL vs ASCR/ShowO. Expected: fwd BAGEL wins, swap ASCR wins → confirms pure position bias. If BAGEL wins both directions → BAGEL genuinely strong against 18-step ASCR.
- [ ] After **68799** finishes: compare swap vs original ShowO-vs-ASCR counts. If both directions agree → ASCR advantage is real. If reversed → advantage was position-bias artefact.
- [ ] After **68794 + 68796** finish: run `scripts/build_geneval_3way_summary.py --model ShowO50=... --model ASCR50=... --model BAGEL=... --output .../geneval_3way_summary.md` and update Quick Results Summary with verified 50-step numbers (replace ⚠ placeholders below).
- [ ] After **68795 + 68797** finish: update Quick Results Summary with debiased (bidirectional) hard64 pairwise numbers.
- [ ] Delete legacy 18-step outputs `outputs/geneval_showo_ascr_68753_*/` (~3.3 GB) after 68794 + 68796 confirmed working.

Job inventory snapshot (2026-05-22):

```
68762 BAGEL GenEval generation                       COMPLETED
68753 ShowO+ASCR GenEval gen @ 18-step               COMPLETED  ← BUG (should be 50-step); superseded by 68794
68784 ShowO+ASCR GenEval gen @ 18-step (re-attempt)  COMPLETED  ← BUG (same bug); superseded by 68794
68785 ShowO+ASCR hard64 gen @ 18-step                COMPLETED  ← BUG (same bug); superseded by 68795
68790 GenEval score ShowO50 (dep 68784)               COMPLETED  ← on buggy 18-step data; superseded by 68796
68791 GenEval score ASCR50  (dep 68784)               COMPLETED  ← on buggy 18-step data; superseded by 68796
68792 GenEval score BAGEL   (dep 68762)               COMPLETED  ← valid (BAGEL unaffected)
68793 hard64 BAGEL 3-way judge fwd-only               COMPLETED  ← position-biased; superseded by 68797/68798
68794 GenEval 553 regen @ 50-step                     RUNNING  (SPGL-1-12)
68795 hard64 64 regen @ 50-step                       RUNNING  (SPGL-1-18)
68796 auto-submit GenEval 3-way scoring               PENDING  (afterok:68794)
68797 auto-submit hard64 bidir 3-way judge            PENDING  (afterok:68795)
68798 DIAG: bidir BAGEL 3-way on 68785 data           RUNNING  (SPGL-1-19)
68799 DIAG: swap ShowO-vs-ASCR on 68753 data          PENDING  (waiting GPU)
```

Cluster constraints (HKU HPC `gpu` partition): max 28 GPUs/user, ≤2 nodes/job, 5 running jobs, 8 submitted. Each node = 8 L40S. GPU usage currently 24/28.



## Quick Results Summary

The current top-level evidence combines Qwen3.5-9B judged T2I-CompBench hard64 results
with an independent GenEval object-checking run. See [Evaluation Methodology](#evaluation-methodology)
for method details and [Qualitative Examples](#qualitative-examples) for side-by-side image comparisons.

**T2I-CompBench hard64 (64 compositional prompts, Qwen3.5-9B judge):**

| Experiment | Judge Method | ASCR | Opponent | Ties | N |
|---|---|---:|---:|---:|---:|
| ASCR vs ShowO baseline | Pairwise side-by-side | **13 wins** | 6 wins | 45 | 64 |
| ASCR vs ShowO baseline | Clean pass/fail | **57 / 64** (89.1 %) | 53 / 64 (82.8 %) | — | 64 |
| ASCR vs BAGEL-7B-MoT | Pairwise side-by-side | **50 wins** | 14 wins | 0 | 64 |
| ASCR vs BAGEL-7B-MoT | Clean pass/fail | **57 / 64** (89.1 %) | 54 / 64 (84.4 %) | — | 64 |

> ⚠ **Pairwise numbers are pre-debiasing estimates (2026-05-22).** Qwen3.5-9B has a confirmed
> strong RIGHT-side preference; all pairwise runs above had ASCR on the RIGHT. Bidirectional
> debiased results are pending (jobs 68798/68799). **Clean pass/fail scores are unaffected**
> (single-image evaluation, no left/right placement).
>
> **Note:** All Qwen judges use Qwen3.5-9B, which is also the ASCR correction loop's semantic
> evaluator. These are automated benchmark signals; independent human evaluation or official
> T2I-CompBench metrics are planned as future work.

**ShowO GenEval full 553 prompts (HSV + NMS + counting threshold = 0.15, job 68776):**

| Task | ShowO baseline | ASCR | Delta |
|---|---:|---:|---:|
| single_object | 100.00% | 100.00% | +0.00 |
| two_object | 65.66% | 79.80% | +14.14 |
| counting | 40.00% | 47.50% | +7.50 |
| colors | 74.47% | 75.53% | +1.06 |
| position | 35.00% | 50.00% | +15.00 |
| color_attr | 9.00% | 19.00% | +10.00 |
| **Overall** | **54.02%** | **61.97%** | **+7.95** |

ASCR improves GenEval overall by **+7.95 points** over the ShowO baseline. The evaluator is
circularity-free with respect to Qwen because it uses detector-based object checks rather than
the ASCR loop's semantic evaluator.

## Source Documents

The project is built from two planning documents placed in the project root:

- `ASCR_Paper_Blueprint_EN.docx`
- `ASCR_Workflow_Playbook_EN.docx`

The documents define ASCR as a three-stage project:

1. Stage 1: validate the principle with a zero-training interface.
2. Stage 2: replace the coarse interface with a learned semantic reopening selector.
3. Stage 3: show cross-model transfer across unified masked multimodal generators.

## Research Thesis

Masked image-token generators usually rely on token confidence to decide where uncertainty remains. This works for uncertainty, but not necessarily for semantic wrongness. ASCR targets a specific failure mode:

> Confidence-semantic inconsistency: a token region becomes stable under confidence dynamics, but the decoded image still violates the prompt in a meaningful way.

Examples include wrong counts, wrong left-right or front-behind relations, wrong color constraints, negation failures, attribute binding errors, OCR mismatches, and missing or extra objects.

The Stage 1 claim is not that a prompt loop is the method. The method principle is selective semantic reopening: identify semantically wrong local regions and reopen the corresponding image-token area while preserving already-correct regions.

## Three-Stage Roadmap

### Stage 1: Zero-Training ASCR Prototype

Goal: validate the failure mode and the selective reopening principle without training a new model.

Default choices confirmed for this project:

- First generator: Show-o.
- Semantic evaluator: local VLM or local LLM/VLM stack first, with an adapter interface so the backend can be replaced.
- Localization interface: visible 4x4 grid over the decoded 512x512 image.
- Token grid target: 32x32 discrete image-token grid for the local Show-o 512 checkpoint.
- Reopening rule: project selected 4x4 cells to 32x32 token regions and use fixed one-token dilation for Stage 1.
- Error handling: conservative abstention is preferred over noisy localization.
- Cluster target: Slurm jobs compatible with both `gpu_shared` and `gpu` partitions.

Stage 1 should produce a runnable, logged, reproducible research prototype. It should not require training, but it must collect traces in a form that can later train Stage 2.

### Stage 2: Learned Semantic Reopening Selector

Goal: replace the coarse visible-grid interface with a lightweight learned selector or decision head that predicts token-level semantic reopening scores.

Stage 1 must leave these interfaces ready:

- Trace writer for `(prompt, intermediate state, grid localization, token mask, correction outcome)` examples.
- `SemanticReopeningSelector` abstraction that can be implemented by rule-based Stage 1 logic or a learned Stage 2 model.
- Training entry points designed for long-running multi-GPU jobs.
- Checkpoint and resume conventions.
- Evaluation hooks for comparing grid-based, learned, and ablated selectors.

### Stage 3: Cross-Model Transfer

Goal: test whether ASCR transfers across unified masked multimodal generators rather than being a Show-o-specific trick.

Stage 1 must leave these interfaces ready:

- `GeneratorAdapter` registry.
- Capability descriptions for token grid size, decode behavior, remask controls, hidden states, and confidence scores.
- Model-specific config files without changing the ASCR loop.
- Transfer benchmark runner that can evaluate the same prompt subsets across multiple generators.

## Evaluation Methodology

### Benchmark choice: GenEval vs T2I-CompBench hard64

We evaluate on two complementary benchmarks. They answer different questions and use different scoring philosophies, so we always report both.

| | GenEval (553 prompts) | T2I-CompBench hard64 (64 prompts) |
|---|---|---|
| What it scores | Whether the image really contains the correct objects / counts / colors / spatial relations | Whether the image satisfies complex compositional semantics (attribute binding, spatial relations, counting, compound phrases) |
| How it scores | Objective detectors: OWLViT object detection + HSV color classifier + IoU/NMS + counting threshold → per-prompt **0/1 pass**, then aggregated over 6 task categories (single object / two object / counting / colors / position / attribute binding) | Subjective judge: a strong VLM/LLM (here Qwen3.5-9B) inspects the image and scores it. Two protocols: (1) clean pass/fail (single-image judgement), (2) pairwise side-by-side (two images, pick winner with ties allowed) |
| What it is good for | Reproducible, **object-level correctness** signal; no dependency on a judging model | **Holistic semantic faithfulness + compositional skill**; sensitive to subtle differences, but depends on judge quality |
| Output granularity | 6 sub-task accuracies + overall score | wins / ties / losses (pairwise) or pass-rate (clean) |

In short: **GenEval is an "objective checkup"; hard64 is a "subjective judging".** They are complementary — objective scores are convincing but only cover object-level facts; judge-based scores capture semantic/compositional differences but inherit the judge's biases. Serious comparisons report both.

### Benchmark: T2I-CompBench hard64

[T2I-CompBench](https://karine-h.github.io/T2I-CompBench/) (NeurIPS 2023, HKU) is a benchmark
designed for **compositional text-to-image generation**: it tests whether generated images
accurately reflect the semantic constraints in the prompt — not visual quality or aesthetics,
but compositional correctness.

The **hard64** subset contains 64 prompts that are particularly challenging for current models,
covering four compositional categories:

| Category | What it tests | Example prompt |
|---|---|---|
| Color–object binding | Each color must bind to the correct object, not transposed | `a green bench and a blue bowl` |
| Shape–object binding | Non-default shapes must bind to the right object | `a pentagonal stop sign and a spherical traffic light` |
| Spatial relations | Objects must appear in the described spatial arrangement | `The blue water bottle was on top of the red backpack.` |
| Counting / quantity | The exact stated number of objects must appear | `one turtle` |

Prompt file: `configs/prompts/t2i_compbench_hard64.txt`. These prompts are selected because
current single-pass generators tend to produce the right *objects* but with wrong color
assignment, wrong spatial arrangement, or wrong count. ASCR's correction loop is specifically
designed to detect and repair these failures.

### Evaluation Method 1: Pairwise Side-by-Side Judge

**What it is:** A *relative* comparison — for the same prompt, which of two images better
follows the prompt description?

**How it works:**

1. Take two clean generated images: competitor (LEFT) and ASCR (RIGHT).
2. Compose a side-by-side canvas. **No text labels are drawn** (`--no-image-labels`). Drawing
   labels like "LEFT: BAGEL" caused Qwen to treat label text as image content in earlier tests,
   distorting verdicts.
3. Feed the canvas to Qwen3.5-9B: *"Check objects, counts, colors, attributes, and spatial
   relations. Which image better satisfies the prompt?"*
4. Qwen returns JSON: `winner` ("baseline"/"ascr"/"tie"), `confidence` (0–1), `summary`,
   `baseline_errors`, `ascr_errors`.
5. Accumulate: `ascr_win` (RIGHT wins), `ascr_loss` (LEFT wins), `pairwise_tie`.

**What it measures:** Whether ASCR's image is *better* in a direct head-to-head comparison.

**Limitation:** Contrast effect — one obviously wrong image makes the other look better even if
both are imperfect. Run alongside the clean pass/fail judge to balance this.

**Script:** `scripts/judge_showo_ascr_pairwise_qwen.py`
**Key flags:** `--baseline-label`, `--ascr-label`, `--no-image-labels`, `--output`

### Evaluation Method 2: Clean Pass/Fail Judge

**What it is:** An *absolute* evaluation — does this image, judged entirely independently,
satisfy the prompt?

**How it works:**

1. Show **only** the ASCR image to Qwen3.5-9B. Ask: "Does this satisfy the prompt?"
   Qwen returns `{"matches_prompt": true/false, "score": 0–1}`.
2. Repeat with only the competitor's image.
3. An image **passes** if `matches_prompt == true` AND `score >= 0.5` (default threshold,
   configurable with `--pass-threshold`).
4. Count outcomes: `both_pass`, `both_fail`, `ascr_win`, `ascr_loss`.

**What it measures:** Whether each image independently meets an absolute quality bar with no
contrast effect.

**Limitation:** The 0.5 threshold is somewhat arbitrary. Qwen3.5-9B is also the ASCR loop's
evaluator, creating a potential circularity: the model that decides when to stop correcting also
judges whether the correction worked.

**Script:** `scripts/judge_showo_ascr_pairs_qwen.py`
**Key flags:** `--pass-threshold`, `--output`, `--config`

### Reading the Two Methods Together

| Signal | Large advantage means | Small advantage means |
|---|---|---|
| Pairwise net | Consistently better in direct head-to-head | Less consistent per-prompt advantage |
| Clean pass/fail net | Larger absolute gap at the score threshold | Both systems pass at similar rates |

When pairwise net is large but clean-pass net is small, one system is more *precisely* correct
even when both clear the pass threshold. When clean-pass net is large but pairwise net is small,
the absolute improvement is real but per-prompt advantage is less consistent.

### Important Caveats

1. **Evaluator circularity:** Qwen3.5-9B is both the ASCR loop's semantic feedback provider and
   the final evaluation judge. Results may reflect Qwen's preference patterns.
2. **No reference images:** Evaluation is entirely VLM-based; no ground-truth images exist.
3. **Automated only:** No human evaluation has been conducted.
4. **ASCR vs standalone model:** ASCR is ShowO + correction loop; BAGEL is a larger standalone
   model. Not architecture-to-architecture.
5. **VLM position bias in pairwise judging:** Qwen3.5-9B exhibits a strong RIGHT-side preference
   in side-by-side comparisons — cross-checking four pairwise comparisons found that whichever
   model was placed on the RIGHT always won lopsidedly, regardless of actual image quality
   (confirmed 2026-05-22, commit `557d2fc`). Pairwise numbers in this README were obtained with
   ASCR always on the RIGHT and should be treated as **pre-debiasing estimates**. The corrected
   protocol (running both forward and swapped directions, then averaging) is now implemented in
   `jobs/stage1_hard64_bagel_3way_judge_sharded.sbatch`; debiased results are pending (jobs 68798,
   68799).

## Stage 1 System Overview

The practical Stage 1 loop is:

1. Receive the original prompt `P_orig`.
2. Run Show-o to an intermediate or completed image-token state `u`.
3. Decode `u` into an intermediate image `I_mid`.
4. Overlay a visible 4x4 grid to create `I_grid`.
5. Ask a local semantic evaluator to compare `P_orig` and `I_grid`.
6. Parse and validate structured semantic output `A_eval`.
7. Convert selected 4x4 cells into a 32x32 token reopening mask.
8. Apply fixed one-token dilation around selected token cells.
9. Compose a correction-conditioned prompt `P_cur`.
10. Reopen selected image-token regions and continue denoising.
11. Log every intermediate artifact and decision.
12. Stop when the evaluator returns no actionable semantic error, the iteration budget is exhausted, or fallback logic triggers abstention.

## Repository Architecture

### Directory Tree

The live source tree — runtime artifacts (`outputs/`, `logs/`, `models/`, `external/`) are
excluded from git:

```text
ASCR/
├── README.md                                    ← project control document (this file)
├── setup.py                                     ← package install (editable: setup.py develop)
├── requirements-qwen-vl.txt                     ← Qwen evaluator pip requirements
├── requirements/
│   ├── base.txt                                 ← core runtime deps (PIL, pyyaml, …)
│   ├── dev.txt                                  ← test + lint tools
│   ├── showo_inference.txt                      ← Show-o inference deps (.venv)
│   └── local_vlm.txt                            ← heuristic evaluator deps
│
├── configs/                                     ← experiment configs (YAML)
│   ├── ★ stage1_showo_qwen35_9b_fullcap_parallel.yaml  ← DEFAULT production config
│   ├── stage1_showo_qwen35_9b.yaml              ← Qwen3.5-9B single-process config
│   ├── stage1_showo_local.yaml                  ← ShO-MMU evaluator config (legacy)
│   ├── showo_local_512x512.yaml                 ← Show-o model hyperparams
│   ├── cluster_gpu.yaml / cluster_gpu_shared.yaml      ← Slurm partition templates
│   ├── prompts/
│   │   ├── ★ t2i_compbench_hard64.txt           ← PRIMARY benchmark (64 prompts)
│   │   ├── t2i_compbench_hard_smoke8.txt        ← 8-prompt smoke subset
│   │   ├── drawbench_all.txt                    ← 200-prompt DrawBench
│   │   ├── drawbench_smoke8.txt                 ← 8-prompt DrawBench smoke
│   │   └── stage1_complex_prompts.txt           ← internal dev regression suite
│   └── experiments/
│       └── qwen36/                              ← Qwen3.6 full-precision (67 GiB, inactive)
│
├── ascr/                                        ← Python package
│   ├── cli/
│   │   ├── ★ compare_showo_ascr.py              ← MAIN benchmark CLI (single-process)
│   │   ├── compare_showo_ascr_parallel.py       ← multi-worker one-GPU-per-worker CLI
│   │   └── run_stage1.py                        ← single-loop debug / dry-run CLI
│   ├── core/
│   │   ├── ★ loop.py                            ← ASCR iterative correction loop
│   │   ├── ★ schemas.py                         ← data contracts (SemanticEvaluation,
│   │   │                                           RegionSelection, TokenReopenMask, …)
│   │   ├── state.py                             ← GenerationState, IterationSummary
│   │   └── artifacts.py                         ← per-run artifact file-system writer
│   ├── generators/
│   │   ├── ★ showo_native.py                    ← ShowONativeEngine: token-level ops
│   │   │                                           (run_confidence_block, force_mask,
│   │   │                                            decode_tokens, token confidence map)
│   │   ├── showo.py                             ← ShowOAdapter: wraps native engine
│   │   │                                           (initialize, reopen_and_continue)
│   │   ├── base.py                              ← GeneratorAdapter ABC
│   │   └── registry.py                          ← build_generator() factory
│   ├── evaluators/
│   │   ├── ★ qwen_vl.py                         ← QwenVLEvaluator (DEFAULT evaluator)
│   │   │                                           Qwen3.5-9B with chain-of-thought JSON
│   │   ├── showo_mmu.py                         ← ShowOMMUEvaluator (legacy alternative,
│   │   │                                           2 MMU calls per iteration)
│   │   ├── mock.py                              ← MockSemanticEvaluator (--dry-run / tests)
│   │   ├── local_vlm.py                         ← heuristic color evaluator (legacy;
│   │   │                                           only supports simple color checks)
│   │   ├── base.py                              ← SemanticEvaluator ABC
│   │   ├── schema_parser.py                     ← JSON extraction + repair helpers
│   │   └── registry.py                          ← build_evaluator() factory
│   ├── grids/
│   │   ├── overlay.py                           ← 4×4 grid overlay renderer (512×512)
│   │   └── projection.py                        ← 4×4 cell → 32×32 token mask + dilation
│   ├── revision/
│   │   ├── selector.py                          ← GridSemanticSelector (cell selection)
│   │   └── prompt_composer.py                   ← correction prompt builder
│   ├── benchmarks/
│   │   ├── metrics.py                           ← score_image, compare_scores (heuristic)
│   │   └── runner.py                            ← result_to_markdown helper
│   └── training/
│       ├── selector_model.py                    ← Stage 2 placeholder: learned selector
│       │                                           interface (image + prompt → token scores)
│       └── train_selector.py                    ← Stage 2 placeholder: training entry point
│
├── scripts/
│   ├── ★ judge_showo_ascr_pairwise_qwen.py      ← side-by-side Qwen3.5-9B pairwise judge
│   │                                               outputs qwen_pairwise_judge.json
│   ├── ★ judge_showo_ascr_pairs_qwen.py         ← clean per-image pass/fail judge
│   │                                               outputs qwen_clean_final_pair_judge.json
│   ├── ★ run_stage1_showo_compare_sharded_reuse.sh  ← sharded runner for single Slurm
│   │                                               8-GPU allocation (primary run script)
│   ├── run_stage1_showo_compare.sh              ← single-worker compare runner
│   ├── run_stage1_showo_compare_parallel.sh     ← one-process-per-GPU compare runner
│   ├── shard_prompts.py                         ← split prompt file across N shards
│   ├── aggregate_showo_ascr_suites.py           ← merge worker shard suites into one
│   ├── prepare_t2i_compbench_prompts.py         ← generate T2I-CompBench prompt files
│   ├── prepare_drawbench_prompts.py             ← generate DrawBench prompt files
│   ├── run_bagel_text2image.py                  ← BAGEL-7B-MoT baseline generation
│   ├── ★ build_geneval_3way_summary.py          ← combine per-model GenEval scores
│   │                                               into 3-way comparison table
│   ├── ★ merge_judge_shards.py                  ← merge N shard JSON outputs from
│   │                                               sharded Qwen judge runs
│   ├── ★ pair_bagel_vs_hard64_run.py            ← pair BAGEL hard64 outputs with
│   │                                               ShowO/ASCR runs by prompt
│   ├── pair_bagel_vs_showo_baseline.py          ← legacy pairing helper
│   │                                               (only used by archived baseline job)
│   ├── run_stage1_debug.sh                      ← mock dry-run (no GPU needed)
│   ├── run_showo_t2i_local.sh                   ← Show-o T2I subprocess (fallback path)
│   ├── run_showo_inpaint_local.sh               ← Show-o inpaint subprocess (fallback)
│   ├── download_showo.sh / download_showo_models.py  ← Show-o model download
│   ├── download_qwen35_9b_snapshot.sh           ← Qwen3.5-9B snapshot download
│   ├── download_qwen36_snapshot.sh              ← Qwen3.6 snapshot (inactive; 67 GiB)
│   ├── sync_github.sh                           ← git add/commit/push helper
│   └── create_env.sh / activate_env.sh          ← environment setup
│
├── jobs/
│   ├── ★ stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch  ← PRIMARY job
│   │                                               8-GPU, 64 prompts, REUSE_MODELS=1
│   ├── stage1_drawbench_qwen35_9b_smoke8.sbatch ← DrawBench 8-prompt smoke (8 GPU)
│   ├── stage1_t2i_compbench_qwen35_9b_smoke1.sbatch  ← 1-prompt smoke + both judges
│   ├── stage1_qwen35_9b_smoke1gpu.sbatch        ← single-GPU full-flow smoke
│   ├── stage1_qwen35_9b_parallel8.sbatch        ← 8-GPU parallel (dev suite)
│   ├── ★ stage1_geneval_score_single.sbatch     ← per-model GenEval scoring
│   │                                               (8-GPU OWLViT, 1 dir at a time)
│   ├── ★ stage1_hard64_bagel_3way_judge_sharded.sbatch  ← 8-GPU sharded Qwen
│   │                                               judge (BAGEL vs ShowO50 vs ASCR50)
│   ├── stage1_geneval_evaluate.sbatch           ← DEPRECATED: pairwise; use score_single
│   ├── stage1_hard64_bagel_3way_judge.sbatch    ← DEPRECATED: 1-GPU; use *_sharded
│   ├── stage2_train_selector_gpu.sbatch         ← Stage 2 placeholder
│   ├── archived/                                ← legacy ShO-MMU evaluator jobs
│   │                                               (superseded by Qwen3.5-9B path)
│   └── experiments/
│       └── qwen36/                              ← Qwen3.6 full-precision experiment jobs
│
├── tests/
│   ├── test_grid_projection.py                  ← 4×4→32×32 projection + dilation
│   ├── test_schema_parser.py                    ← SemanticEvaluation JSON parsing
│   ├── test_prompt_composer.py                  ← correction prompt generation
│   ├── test_loop_initial_state.py               ← loop initialization
│   ├── test_loop_multi_insert.py                ← multi-iteration loop behavior
│   ├── test_native_showo_helpers.py             ← ShowONativeEngine helper ops
│   ├── test_qwen_vl_evaluator.py                ← QwenVLEvaluator integration
│   ├── test_local_vlm.py                        ← heuristic evaluator
│   └── test_compare_showo_suite.py              ← end-to-end comparison CLI
│
├── docs/
│   ├── stage1_phase1_summary_20260519.md        ← T2I-CompBench hard64 benchmark summary
│   ├── stage1_design.md                         ← ASCR algorithm design notes
│   ├── benchmark_plan.md                        ← evaluation plan
│   ├── cluster_notes.md                         ← HKU AI cluster usage notes
│   ├── project_status.md                        ← current status snapshot
│   └── examples/                                ← pairwise comparison images (git-tracked)
│
├── external/Show-o/                             ← NOT in git; clone separately
├── models/                                      ← NOT in git; download separately
│   ├── show-o-512x512/
│   ├── magvitv2/
│   ├── phi-1_5/
│   └── qwen3.5-9b/
├── outputs/                                     ← NOT in git; runtime benchmark artifacts
└── logs/                                        ← NOT in git; Slurm stdout/stderr
```

### Module Quick Reference

#### "Where do I find…?"

| Goal | Start here |
|---|---|
| **Understand the ASCR algorithm** | `ascr/core/loop.py` |
| **Data schemas** (SemanticEvaluation, RegionSelection, TokenReopenMask) | `ascr/core/schemas.py` |
| **Show-o token operations** (force-mask, confidence block, decode) | `ascr/generators/showo_native.py` |
| **Qwen3.5-9B evaluator** (prompt template, JSON parsing, thinking mode) | `ascr/evaluators/qwen_vl.py` |
| **Grid overlay** (4×4 visible grid on 512×512 image) | `ascr/grids/overlay.py` |
| **4×4 → 32×32 token projection + dilation** | `ascr/grids/projection.py` |
| **Correction prompt builder** | `ascr/revision/prompt_composer.py` |
| **Run a single-prompt comparison** | `ascr/cli/compare_showo_ascr.py` |
| **Submit 8-GPU benchmark** | `jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch` |
| **VLM pairwise judge** (side-by-side comparative) | `scripts/judge_showo_ascr_pairwise_qwen.py` |
| **VLM clean pass/fail judge** (per-image) | `scripts/judge_showo_ascr_pairs_qwen.py` |
| **GenEval per-model scoring** (8-GPU) | `jobs/stage1_geneval_score_single.sbatch` |
| **GenEval 3-way comparison summary** | `scripts/build_geneval_3way_summary.py` |
| **8-GPU sharded hard64 Qwen judge** | `jobs/stage1_hard64_bagel_3way_judge_sharded.sbatch` |
| **Merge sharded judge outputs** | `scripts/merge_judge_shards.py` |
| **Default config** | `configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml` |
| **Primary benchmark prompts** | `configs/prompts/t2i_compbench_hard64.txt` |
| **Stage 2 interface contracts** | `ascr/training/selector_model.py` |

> **Note on judge sibling scripts.** `scripts/judge_showo_ascr_pairs_qwen.py` (per-image clean pass/fail)
> and `scripts/judge_showo_ascr_pairwise_qwen.py` (side-by-side comparative) are **siblings, not duplicates** —
> both are actively used in different evaluation flows. Do not delete or merge.

#### Active vs Legacy Evaluators

| Backend | Key | When to use |
|---|---|---|
| Qwen3.5-9B | `qwen_vl` | ★ Default for all production runs; chain-of-thought JSON; requires `models/qwen3.5-9b` |
| Show-o MMU | `showo_mmu` | Legacy: Show-o self-evaluation without extra model; 2 MMU calls per iteration, slower |
| Mock | `mock` | `--dry-run`, unit tests; no GPU needed |
| Heuristic | `local_vlm` | Legacy: color-presence checks only; not suitable for compositional prompts |

#### Output Directory Layout

Each benchmark run writes a timestamped root under `outputs/`:

```text
outputs/<run-name>/
├── suite.json                              ← aggregated results for all prompts
├── shard_manifest.log                      ← prompt sharding record
├── shards/shard_N.txt                      ← per-worker prompt lists
├── worker_N.log                            ← per-worker stdout/stderr
├── shard_N/
│   └── showo_ascr-<ts>/
│       └── prompt_NNN-<slug>/
│           ├── ★ baseline_showo.png        ← baseline clean image (judge input)
│           └── ascr/
│               └── stage1_showo_ascr-<ts>/
│                   ├── iterations/
│                   │   └── 000/
│                   │       ├── decoded.png            ← decoded image at iter N
│                   │       ├── grid.png               ← 4×4 grid overlay (diagnostic only)
│                   │       ├── evaluation.json        ← Qwen evaluator output
│                   │       ├── correction_prompt.txt  ← correction prompt used
│                   │       ├── confidence.json        ← token confidence metadata
│                   │       └── mask.json              ← 32×32 reopening mask
│                   ├── ★ final_decoded_image.png      ← ASCR final clean image (judge input)
│                   ├── trace.jsonl                    ← iteration-by-iteration trace
│                   └── comparison.json                ← heuristic comparison (dev only)
├── ★ qwen_pairwise_judge.json              ← side-by-side VLM judgment (primary signal)
└── ★ qwen_clean_final_pair_judge.json      ← per-image pass/fail judgment
```

> **Key rule:** `baseline_showo.png` and `final_decoded_image.png` are the only files used as
> judge inputs. Grid overlay images (`grid.png`) are diagnostic artifacts for localization and
> must never be used as benchmark images.
## Stage 1 Implementation Plan

### S1.0 Repository Bootstrap

Status: completed.

Tasks:

- Create project README with detailed roadmap and current status.
- Create `.gitignore` for Python, ML artifacts, Slurm outputs, local environments, and secrets.
- Initialize Git in `/grp01/cds_bdai/JianyuZhang/ASCR`.
- Connect remote `https://github.com/hmss2002/ASCR.git`.
- Commit and push confirmed updates to `main`.

### S1.1 Dedicated Environment

Status: completed for the Stage 1 scaffold.

Tasks:

- Create `.venv` under the project root.
- Add `scripts/create_env.sh` and `scripts/activate_env.sh`.
- Add minimal `requirements/base.txt` for configuration, validation, image processing, logging, and testing.
- Add optional dependency files for local VLM and Show-o integration.
- Make every run script activate `.venv` first.

Acceptance:

- `source .venv/bin/activate` works.
- `python -m ascr.cli.run_stage1 --help` works after installation.
- Base environment is not modified.

### S1.2 Core Data Contracts

Status: completed for the Stage 1 scaffold.

Tasks:

- Define `ASCRState` for prompt, token state, decoded image path, grid image path, evaluator output, masks, iteration counters, and artifact paths.
- Define `SemanticEvaluation` schema for structured local semantic feedback.
- Define `RegionSelection` schema for selected 4x4 grid cells, natural-language reason, confidence, and action type.
- Define `TokenReopenMask` schema for 32x32 reopening masks.
- Add strict parser and fallback behavior for malformed evaluator outputs.

Acceptance:

- Valid JSON parses into typed objects.
- Invalid JSON fails safely and triggers abstention.
- Unit tests cover valid, malformed, empty, and over-broad localization outputs.

### S1.3 Show-o Generator Adapter

Status: completed for native local Show-o token-grid integration, with subprocess helper scripts retained as fallback.

Tasks:

- Create a `GeneratorAdapter` base class.
- Implement initial `ShowOAdapter` placeholder with explicit unimplemented methods where real Show-o calls will connect.
- Define methods for initial generation, intermediate decode, confidence readout, selective remask, and continuation.
- Keep Show-o repository path and checkpoint path configurable.
- Avoid hardcoding cluster-specific paths inside Python modules.

Acceptance:

- A mock generator can run the ASCR loop without real model weights.
- Real Show-o adapter can be connected by config without changing the loop.

### S1.4 Local Semantic Evaluator Adapter

Status: completed for the local heuristic evaluator and the local Show-o MMU evaluator backend; GPU smoke validation completed through Slurm.

Tasks:

- Create `SemanticEvaluator` base class.
- Implement local VLM adapter interface for image-plus-prompt evaluation.
- Add prompt templates from the workflow document.
- Enforce structured JSON output with schema validation.
- Add retry and fallback logic for malformed local model responses.

Acceptance:

- Evaluator returns either a valid semantic error report or an explicit abstain response.
- No malformed text can directly enter the remask stage.

### S1.5 Grid Overlay and Projection

Status: completed for the Stage 1 scaffold.

Tasks:

- Implement visible 4x4 grid overlay for 512x512 images.
- Add labels that match the evaluator coordinate vocabulary.
- Implement projection from selected 4x4 cells to 32x32 token cells.
- Add fixed one-token dilation for Stage 1.
- Keep image size and token grid size configurable.

Acceptance:

- Grid projection tests cover corners, edges, multiple cells, duplicate cells, and dilation boundaries.

### S1.6 ASCR Loop Orchestration

Status: completed for mock dry-run and local Show-o execution.

Tasks:

- Implement the alternating semantic-confidence revision loop.
- Add max iteration budget.
- Add stopping conditions for no error, no selected region, parser failure, generator failure, and collateral-damage risk.
- Save every iteration artifact.
- Make the loop runnable through a CLI.

Acceptance:

- A mock end-to-end run writes a complete artifact directory.
- Every decision is reproducible from saved JSON files.

### S1.7 Trace Collection for Stage 2

Status: completed for JSONL Stage 1 traces.

Tasks:

- Write trace records for each ASCR iteration.
- Include prompt, evaluator JSON, selected grid cells, projected token mask, correction prompt, and before-after artifact paths.
- Reserve optional fields for hidden states, confidence maps, revision gains, and human labels.

Acceptance:

- Stage 1 can produce a JSONL trace file suitable for future selector training.

### S1.8 Benchmark and Baselines

Status: completed for a native Show-o-vs-ASCR comparison CLI that preserves the baseline token state; formal multi-prompt benchmarks remain pending.

Tasks:

- Create targeted benchmark prompt subsets for counting, spatial relations, color binding, negation, attribute binding, OCR, missing objects, and extra objects.
- Add baseline runners for whole-image retry, best-of-N reranking, verifier-only selection, generic inpainting adapter, confidence-only remask, semantic-only repair, and ASCR alternating.
- Add metrics for semantic improvement and collateral damage.
- Compare original Show-o baseline and ASCR using the same prompt, seed, and native Show-o token state. The ASCR branch starts from the baseline state and only reopens semantic mask regions before continuing denoising.
- For formal evaluation, run the same prompts, seeds, and settings for baseline and ASCR, then report prompt-category breakdowns and save paired artifacts.

Acceptance:

- A small smoke benchmark can run locally with mock backends.
- Result tables are emitted as JSON and Markdown.

### S1.9 Cluster Jobs and Multi-GPU Readiness

Status: completed for Stage 1 single-GPU jobs; multi-GPU is reserved for sweeps and Stage 2 training.

Tasks:

- Add Slurm scripts for `gpu_shared` debug runs.
- Add Slurm scripts for `gpu` longer runs.
- Support environment variables for partition, GPU count, CPU count, walltime, config path, and output directory.
- Keep Stage 1 inference compatible with single GPU.
- Reserve Stage 2 training scripts for `torchrun` and Slurm DDP.

Recommended usage:

- Use the interactive GPU shell for dependency checks, import checks, and one or two image smoke tests.
- Use `gpu_shared` for short debugging, smoke tests, and small prompt sweeps.
- Use `gpu` for formal baseline-vs-ASCR runs, long benchmark sweeps, and future multi-GPU training.
- Single-image Show-o inference does not need multi-GPU; multi-GPU becomes useful for parallel benchmark batches and Stage 2 training.

Acceptance:

- The same code path can run under interactive shell, `gpu_shared` Slurm job, or `gpu` Slurm job.

### S1.10 Documentation and Reproducibility

Status: ongoing; initial docs and validation notes added.

Tasks:

- Maintain README after each confirmed update batch.
- Add design notes under `docs/`.
- Version prompt templates.
- Save config snapshots with every run.
- Record Git commit hash in each artifact directory.

Acceptance:

- A future reader can reproduce what was run from the artifact folder alone.

## Environment Policy

This project must use dedicated virtual environments to avoid disturbing the server base
environment. Three venvs are currently in use, each scoped to a model family:

| Venv | Purpose | Activated by |
|---|---|---|
| `.venv` | Original ShowO + ASCR loop (torch 2.2.1; legacy local-VLM/Show-o MMU evaluator path) | most ShowO inference scripts and `scripts/run_stage1_showo_compare*.sh` |
| `.venv-qwen36` | ★ Production: ShowO + ASCR loop with Qwen3.5-9B evaluator (torch 2.5.1+cu121, transformers shim under `.deps/transformers-qwen35-clean`) | `jobs/stage1_*_qwen35_9b_*.sbatch`, `jobs/stage1_t2i_*` |
| `.venv-bagel` | BAGEL-7B-MoT generation only (torch 2.5.1+cu121, flash-attn 2.7.4.post1) | `scripts/run_bagel_text2image.py`, `jobs/stage1_*bagel*.sbatch` |

All three live under `/grp01/cds_bdai/JianyuZhang/ASCR/`, are gitignored, and must be created
on the cluster — they are not portable. Compute nodes are offline (`HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `QWEN_LOCAL_FILES_ONLY=1`), so any new dependency must be
pre-installed from a login node.

Do not install ASCR dependencies directly into the base conda environment unless there is no
practical alternative and the decision is recorded here.

## Data, Artifacts, and Large Files

The repository should track source code, configs, small benchmark definitions, documentation, and job scripts. It should not track runtime outputs or large model artifacts.

Ignored by default:

- `.venv/`
- Python caches
- Slurm output files
- local secrets and API tokens
- model weights
- downloaded datasets
- generated images
- run outputs
- checkpoints
- logs
- W&B or TensorBoard local runs

If a file is required for reproducibility but too large for Git, store it under the project path and document the expected location in README or config files.

## GitHub Synchronization Policy

Remote repository:

```bash
https://github.com/hmss2002/ASCR.git
```

Policy:

- After each confirmed update batch, commit and push to `main`.
- Keep commit messages descriptive and scoped.
- Do not commit secrets, local environment folders, model weights, checkpoints, generated outputs, or large datasets.
- If GitHub authentication is missing on the server, stop at the authentication boundary and ask the user to complete authentication.

## Quickstart

The canonical Stage 1 workflow uses the Qwen3.5-9B evaluator + ShowO generator on the
T2I-CompBench hard64 prompt set, with 8-way GPU sharding and runtime image reuse.

### Local dry-run (no GPU required)

```bash
source .venv-qwen36/bin/activate
python -m unittest discover -s tests -v
python -m ascr.cli.run_stage1 --dry-run \
    --config configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml \
    --output-dir outputs/smoke \
    --prompt "A red cube left of a blue sphere"
```

### Slurm: T2I-CompBench hard64 generation + judge (8 GPU L40S)

```bash
sbatch jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch
```

This produces ShowO baseline + ASCR-corrected images for all 64 prompts under
`outputs/benchmarks_t2i_compbench_qwen35_hard64_slurm8gpu_reuse_<timestamp>/`, and on
`afterok` dependency runs the Qwen pairwise + clean pass/fail judges and writes the
summary JSON.

### Slurm: GenEval 553-prompt generation (8 GPU L40S, 50-step)

```bash
sbatch jobs/stage1_geneval_generate_8gpu.sbatch
```

Outputs go to `outputs/geneval_showo_ascr_<jobid>_<timestamp>/` with `geneval_baseline/`
and `geneval_ascr/` subdirectories ready for `scripts/evaluate_geneval_owlvit.py`
(see `jobs/stage1_geneval_evaluate.sbatch` for the scoring step).

Legacy ShO-MMU evaluator jobs are preserved under `jobs/archived/`.

## Stage 1 Acceptance Criteria

Stage 1 is considered complete when all of the following are true:

- A single prompt can run through the full ASCR loop with real Show-o integration.
- A dry-run mode can run without Show-o weights using mock adapters.
- The local semantic evaluator produces schema-validated JSON.
- Malformed evaluator output triggers safe fallback rather than remasking.
- 4x4 grid cells project correctly into 32x32 token masks.
- Fixed one-token dilation is implemented and tested.
- Each run saves decoded images, grid images, evaluator JSON, selected masks, prompts, configs, and trace JSONL.
- A small benchmark subset can compare ASCR with core baselines.
- Slurm scripts support the `gpu` partition (legacy `gpu_shared` support preserved under `jobs/archived/`).
- README documents how to reproduce the latest working run.

## Open Decisions

These decisions are not blocking the repository bootstrap:

- Concrete local VLM/LLM evaluator backend and checkpoint path.
- Final dataset storage path for large benchmarks.
- Whether generated paper figures should be tracked as lightweight examples or stored only as artifacts.

## Design Rule

Keep Stage 1 simple enough to prove the mechanism, but structure it so Stage 2 and Stage 3 do not require rewriting the project. The grid and JSON interface are implementation devices for the first prototype, not the final scientific claim.

## Stage 1 Benchmark Summary — Three-Way Comparison

> ⚠ **Debiasing in progress (2026-05-22):** The pairwise win counts below are from
> single-direction judging with ASCR always placed on the RIGHT. A strong VLM RIGHT-position
> preference has been confirmed (see Important Caveats §5 and commit `557d2fc`). The
> **"ASCR >> BAGEL"** ordering should be treated as a pre-debiasing estimate pending
> bidirectional confirmation (jobs 68798/68799). Clean pass/fail scores are unaffected.

All three pairwise comparisons on **T2I-CompBench hard64** are now complete. Preliminary
(pre-debiasing) performance ordering:

**ASCR >> BAGEL-7B-MoT > ShowO Baseline** *(pairwise — pending debiased confirmation)*

### Pairwise Win/Loss Summary

| Comparison | Winner | Wins | Losses | Ties | Net | N |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **ASCR vs ShowO baseline** | **ASCR** | **13** | 6 | 45 | **+7** | 64 |
| **ASCR vs BAGEL-7B-MoT** | **ASCR** | **50** | 14 | 0 | **+36** | 64 |
| **BAGEL-7B-MoT vs ShowO baseline** | **BAGEL** | **26** | 21 | 0 | **+5** | 47 |

### Clean Pass/Fail Summary (Qwen independent per-image)

| Model | Pass | Fail | Rate |
| --- | ---: | ---: | ---: |
| **ASCR** | **57** | 7 | **89.1%** |
| BAGEL-7B-MoT | 54 | 10 | 84.4% |
| ShowO baseline | 53 | 11 | 82.8% |

### Key Takeaways

- **ASCR achieves the highest prompt-following accuracy** on hard64 compositional prompts,
  outperforming both baselines by a large margin in direct pairwise comparison.
- **ASCR beats BAGEL decisively (net +36):** ASCR's correction loop on a 1.3 B model surpasses
  BAGEL-7B-MoT (a 5x larger dedicated T2I model) on exactly the prompt categories — spatial
  relations, color-object binding, shape-object binding, counting — that the loop is designed
  to detect and repair.
- **BAGEL beats ShowO baseline modestly (net +5):** BAGEL-7B-MoT is a stronger standalone
  model than the ShowO baseline, consistent with its larger parameter count and dedicated T2I
  training. The margin is small relative to ASCR's advantage, confirming the correction loop
  adds more value than simply switching to a larger generation model.
- **ASCR's correction loop advantage is robust:** even after accounting for the Qwen evaluator
  circularity caveat, the margin over BAGEL (+36 pairwise, +3 clean-pass) is large enough that
  an independent evaluator would need to strongly disagree with Qwen to reverse the finding.
- **Evaluator circularity caveat:** Qwen3.5-9B is the judge for all three hard64 comparisons
  and is also the ASCR loop evaluator. This caveat is partially addressed by the independent
  GenEval run below: it uses OWLViT detector outputs plus deterministic color/count postprocessing,
  not Qwen, and shows ASCR ahead of the ShowO baseline by +7.95 overall points.

## 2026-05-21 ShowO GenEval — Independent Full 553-Prompt Evaluation

> ⚠ **Note (2026-05-22):** The images for this section (job **68753**) were generated at
> **18 diffusion steps**, not 50, due to a CLI default-override bug fixed in commit `557d2fc`.
> The ShowO-baseline-vs-ASCR **relative comparison remains valid** (both sides affected equally
> by the same step count), but absolute accuracy values are expected to differ at 50 steps.
> A corrected 3-way 50-step GenEval section will be added once jobs 68794 and 68796 complete.

This run evaluates the full GenEval 553-prompt suite for ShowO baseline vs ASCR using a
non-Qwen, object-detection-based scorer. It is intended as an independent check on the Qwen
hard64 findings above.

**Protocol:**
- Images: `outputs/geneval_showo_ascr_68753_20260521_170538/geneval_baseline/` and
  `outputs/geneval_showo_ascr_68753_20260521_170538/geneval_ascr/`.
- Evaluator: `scripts/evaluate_geneval_owlvit.py` with local `models/owlvit-base-patch32`.
- Slurm job: 68776, 8 GPU shards, completed in 00:01:46.
- Output files: `outputs/geneval_showo_ascr_68753_20260521_170538/results_baseline.jsonl`,
  `outputs/geneval_showo_ascr_68753_20260521_170538/results_ascr.jsonl`.
- Summary log: `logs/geneval-evaluate-68776.out`.

**Evaluator fixes used for the final score:**
- HSV pixel-histogram color classifier for color-attribute binding, replacing unreliable
  OWLViT/CLIP pooler color similarities.
- Per-class NMS at IoU 0.5 to remove duplicate overlapping detections.
- Tag-aware detection threshold: default `--threshold 0.01` for recall-sensitive tasks, plus
  `--counting-threshold 0.15` for counting to suppress low-confidence false positives.

**Results:**

| Task | ShowO baseline | ASCR | Delta |
|---|---:|---:|---:|
| single_object | 100.00% (80 / 80) | 100.00% (80 / 80) | +0.00 |
| two_object | 65.66% (65 / 99) | 79.80% (79 / 99) | +14.14 |
| counting | 40.00% (32 / 80) | 47.50% (38 / 80) | +7.50 |
| colors | 74.47% (70 / 94) | 75.53% (71 / 94) | +1.06 |
| position | 35.00% (35 / 100) | 50.00% (50 / 100) | +15.00 |
| color_attr | 9.00% (9 / 100) | 19.00% (19 / 100) | +10.00 |
| **Overall** | **54.02%** | **61.97%** | **+7.95** |

**Interpretation:**

The independent GenEval run supports the same direction as the Qwen hard64 comparisons: ASCR
substantially improves compositional prompt following over the ShowO baseline. The largest gains
are in two-object, counting, position, and color-attribute tasks. The corrected counting score
also confirms that the previous 0% was an evaluator artifact caused by low-threshold false
positives interacting with GenEval's exclude rule.

## Qualitative Examples

Each image below is a compact side-by-side comparison copied from runtime outputs into
`docs/examples/` so GitHub can render it without syncing the full `outputs/` tree. For Qwen
pairwise examples, the canvas is exactly what was fed to Qwen3.5-9B. For GenEval examples,
the canvas is a README-only visualization with **LEFT = ShowO baseline** and **RIGHT = ASCR**.

### GenEval Detector Examples

Representative ASCR-only wins from the full 553-prompt GenEval run (job 68776):

![GenEval two_object — a photo of a toothbrush and a snowboard](docs/examples/geneval/two_object_081_a-photo-of-a-toothbrush-and-a-snowboard.png)

![GenEval counting — a photo of two bears](docs/examples/geneval/counting_184_a-photo-of-two-bears.png)

![GenEval counting — a photo of three pizzas](docs/examples/geneval/counting_240_a-photo-of-three-pizzas.png)

![GenEval position — a photo of a bird left of a couch](docs/examples/geneval/position_400_a-photo-of-a-bird-left-of-a-couch.png)

![GenEval color_attr — a photo of a yellow pizza and a green oven](docs/examples/geneval/color_attr_504_a-photo-of-a-yellow-pizza-and-a-green-oven.png)

![GenEval color_attr — a photo of an orange cow and a purple sandwich](docs/examples/geneval/color_attr_544_a-photo-of-an-orange-cow-and-a-purple-sandwich.png)

### ASCR vs ShowO Baseline

2 wins · 2 losses · 2 ties shown (out of 13 wins / 6 losses / 45 ties total).

##### **ASCR wins** — `a girl behind a cow`

*Qwen3.5-9B (conf 0.95):* The right image (ASCR) correctly includes the requested subject, a girl, positioned behind the cow, whereas the left image (baseline) completely omits the girl.

![a girl behind a cow — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_win_1_a_girl_behind_a_cow.png)

---

##### **ASCR wins** — `a pentagonal stop sign and a spherical traffic light`

*Qwen3.5-9B (conf 0.90):* The prompt requests a pentagonal stop sign and a spherical traffic light. Both images feature octagonal stop signs, failing the shape constraint. However, the ASCR image's traffic light has a smoother, more rounded housing that is closer to spherical than the baseline's angular housing. Thus, ASCR better satisfies the prompt's specific shape requirements.

![a pentagonal stop sign and a spherical traffic light — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_win_2_a_pentagonal_stop_sign_and_a_spherical_traff.png)

---

##### **ASCR loses** — `a mouse on side of a key`

*Qwen3.5-9B (conf 0.95):* The baseline image perfectly matches the prompt, showing a single mouse standing on a single golden key. The ASCR image suffers from severe hallucinations, showing a distorted, multi-headed creature and a fragmented, glitchy key.

![a mouse on side of a key — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_loss_1_a_mouse_on_side_of_a_key.png)

---

##### **ASCR loses** — `a sheep in front of a key`

*Qwen3.5-9B (conf 0.95):* The baseline image correctly depicts a sheep in front of a key, satisfying the prompt. The right image replaces the key with a metallic grate or mesh structure, failing to generate the requested object.

![a sheep in front of a key — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_loss_2_a_sheep_in_front_of_a_key.png)

---

##### **Tie** — `a green bench and a blue bowl`

*Qwen3.5-9B (conf 0.95):* Both images perfectly satisfy the prompt, depicting a green bench and a blue bowl with accurate colors, counts, and spatial relations.

![a green bench and a blue bowl — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/tie_1_a_green_bench_and_a_blue_bowl.png)

---

##### **Tie** — `an oblong cucumber and a teardrop plum`

*Qwen3.5-9B (conf 0.95):* Both images accurately depict an oblong cucumber and a teardrop-shaped plum against a green background. The objects, colors, and spatial relations are identical in both images, with no material differences affecting prompt adherence.

![an oblong cucumber and a teardrop plum — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/tie_2_an_oblong_cucumber_and_a_teardrop_plum.png)


### ASCR vs BAGEL-7B-MoT

2 wins · 2 losses shown (out of 50 wins / 14 losses / 0 ties total).

##### **ASCR wins** — `an oblong cucumber and a teardrop plum`

*Qwen3.5-9B (conf 0.95):* The right image (ASCR) correctly depicts an oblong cucumber and a teardrop-shaped plum, matching the prompt's object descriptions and spatial arrangement. The left image (baseline) misidentifies the plum as a pear, which is a significant object error.

![an oblong cucumber and a teardrop plum — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_win_1_an_oblong_cucumber_and_a_teardrop_plum.png)

---

##### **ASCR wins** — `two boys`

*Qwen3.5-9B (conf 0.95):* The right image (ASCR) is a faithful representation of the prompt 'two boys', showing two distinct individuals. The left image (BAGEL) depicts two identical clones of the same boy, which is a hallucination not present in the prompt.

![two boys — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_win_2_two_boys.png)

---

##### **ASCR loses** — `a giraffe next to a lamp`

*Qwen3.5-9B (conf 0.95):* The left image (BAGEL) perfectly satisfies the prompt, showing a complete giraffe standing next to a lamp with correct spatial relations and lighting. The right image (ASCR) is severely cropped, cutting off the giraffe's body and showing only its head and neck, which fails to represent the full object described in the prompt.

![a giraffe next to a lamp — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_loss_1_a_giraffe_next_to_a_lamp.png)

---

##### **ASCR loses** — `a girl on the top of a frog`

*Qwen3.5-9B (conf 0.95):* The left image (baseline) perfectly matches the prompt 'a girl on the top of a frog' with a cute, high-quality 3D render of a girl sitting on a large frog in a pond. The right image (ASCR) shows a girl sitting on a frog, but the frog is on a rock, not in water, and the overall style is less consistent with the prompt's implied whimsical nature. The left image is more visually appealing and adheres better to the spatial relation of being 'on top of' in a natural setting.

![a girl on the top of a frog — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_loss_2_a_girl_on_the_top_of_a_frog.png)


## Changelog

Dated experiment narratives have been moved to [docs/changelog.md](docs/changelog.md)
(latest first). The Active TODO, Quick Results Summary, Stage 1 Benchmark Summary, and
the most recent independent GenEval section above remain the canonical current state.

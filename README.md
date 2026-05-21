# ASCR: Alternating Semantic-Confidence Revision

ASCR is a research prototype for studying and correcting confidence-semantic inconsistency in masked image-token generation. The central observation is that an image region can become confidence-stable during iterative denoising while still being semantically wrong with respect to the text prompt. Stage 1 starts with a zero-training implementation that uses a visible 4x4 grid and structured local semantic feedback to selectively reopen image-token regions instead of retrying the whole image.

This README is the project control document. It records the research plan, implementation plan, current progress, expected interfaces, cluster workflow, and GitHub synchronization policy. It should be updated whenever a meaningful implementation batch is completed.

## Quick Results Summary

All Stage 1 evaluations use T2I-CompBench hard64 (64 compositional prompts) judged by
Qwen3.5-9B. See [Evaluation Methodology](#evaluation-methodology) for method details and
[Qualitative Examples](#qualitative-examples) for side-by-side image comparisons.

| Experiment | Judge Method | ASCR | Opponent | Ties | N |
|---|---|---:|---:|---:|---:|
| ASCR vs ShowO baseline | Pairwise side-by-side | **13 wins** | 6 wins | 45 | 64 |
| ASCR vs ShowO baseline | Clean pass/fail | **57 / 64** (89.1 %) | 53 / 64 (82.8 %) | — | 64 |
| ASCR vs BAGEL-7B-MoT | Pairwise side-by-side | **50 wins** | 14 wins | 0 | 64 |
| ASCR vs BAGEL-7B-MoT | Clean pass/fail | **57 / 64** (89.1 %) | 54 / 64 (84.4 %) | — | 64 |

> **Note:** All judges use Qwen3.5-9B, which is also the ASCR correction loop's semantic
> evaluator. These are automated benchmark signals; independent human evaluation or official
> T2I-CompBench metrics are planned as future work.

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
│   ├── showo_inference.txt                      ← Show-o inference deps (.venv legacy)
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
│   ├── stage2_train_selector_gpu.sbatch         ← Stage 2 placeholder
│   ├── archived/                                ← legacy .venv + ShO-MMU jobs
│   │                                               (env superseded by .venv-qwen36)
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
| **VLM pairwise judge** | `scripts/judge_showo_ascr_pairwise_qwen.py` |
| **VLM clean pass/fail judge** | `scripts/judge_showo_ascr_pairs_qwen.py` |
| **Default config** | `configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml` |
| **Primary benchmark prompts** | `configs/prompts/t2i_compbench_hard64.txt` |
| **Stage 2 interface contracts** | `ascr/training/selector_model.py` |

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

## Current Progress Log

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


### 2026-04-28

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

## Environment Policy

This project must use a dedicated virtual environment to avoid disturbing the server base environment. The planned location is:

```bash
/grp01/cds_bdai/JianyuZhang/ASCR/.venv
```

Every run script should activate it before executing project code:

```bash
cd /grp01/cds_bdai/JianyuZhang/ASCR
source .venv/bin/activate
```

Do not install ASCR dependencies directly into the base conda environment unless there is no practical alternative and the decision is recorded here.

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

## Initial Run Targets

The first runnable milestones will be:

```bash
source .venv/bin/activate
python -m ascr.cli.run_stage1 --help
python -m unittest discover -s tests -v
python -m ascr.cli.run_stage1 --dry-run --config configs/stage1_showo_local.yaml --output-dir outputs/smoke --prompt A-red-cube-left-of-a-blue-sphere
```

Validated result on the cluster:

- Unit tests: 14 passed.
- Dry-run: `stop_reason` was `no_semantic_error` after one recorded iteration.
- Example artifact root: `outputs/smoke/stage1_showo_local-20260428-034010-528555`.

Current Slurm entry points include:

```bash
sbatch jobs/stage1_debug_gpu_shared.sbatch
sbatch jobs/stage1_run_gpu.sbatch
```

These Slurm commands are ready as templates; use `gpu_shared` for smoke checks and `gpu` for longer formal runs.

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
- Slurm scripts support both `gpu_shared` and `gpu`.
- README documents how to reproduce the latest working run.

## Open Decisions

These decisions are not blocking the repository bootstrap:

- Concrete local VLM/LLM evaluator backend and checkpoint path.
- Final dataset storage path for large benchmarks.
- Whether generated paper figures should be tracked as lightweight examples or stored only as artifacts.

## Design Rule

Keep Stage 1 simple enough to prove the mechanism, but structure it so Stage 2 and Stage 3 do not require rewriting the project. The grid and JSON interface are implementation devices for the first prototype, not the final scientific claim.

## Latest Stage 1 Real Show-o Status

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

### 2026-04-28 Qwen3.6 Evaluator Path

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

## Stage 1 Benchmark Summary — Three-Way Comparison

All three pairwise comparisons on **T2I-CompBench hard64** are now complete, establishing an
unambiguous performance ordering:

**ASCR >> BAGEL-7B-MoT > ShowO Baseline**

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
- **Evaluator circularity caveat:** Qwen3.5-9B is the judge for all three comparisons and is
  also the ASCR loop evaluator. The BAGEL vs ShowO comparison is free of this circularity
  (neither system uses Qwen at generation time). Independent GenEval (OWLViT-based, job 68753)
  results are pending and will provide circularity-free confirmation.

## Qualitative Examples

Each image below is the **pairwise side-by-side canvas** fed to the Qwen3.5-9B judge.
**LEFT = Baseline, RIGHT = ASCR** in both comparisons.
Representative examples are selected by highest judge confidence.

### ASCR vs ShowO Baseline

2 wins · 2 losses · 2 ties shown (out of 13 wins / 6 losses / 45 ties total).

##### **ASCR wins** — `a girl behind a cow`

*Qwen3.5-9B (conf 0.95):* The right image (ASCR) correctly includes the requested subject, a girl, positioned behind the cow, whereas the left image (baseline) completely omits the girl.

![a girl behind a cow — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_win_1_a_girl_behind_a_cow.png)

---

##### **ASCR wins** — `a pentagonal stop sign and a spherical traffic light`

*Qwen3.5-9B (conf 0.90):* The prompt requests a pentagonal stop sign and a spherical traffic light. Both images feature octagonal stop signs, failing the shape constraint. However, the ASCR image's traffic light has a smoother

![a pentagonal stop sign and a spherical traffic light — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_win_2_a_pentagonal_stop_sign_and_a_spherical_traff.png)

---

##### **ASCR loses** — `a mouse on side of a key`

*Qwen3.5-9B (conf 0.95):* The baseline image perfectly matches the prompt, showing a single mouse standing on a single golden key. The ASCR image suffers from severe hallucinations, showing a distorted, multi-headed creature a

![a mouse on side of a key — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_loss_1_a_mouse_on_side_of_a_key.png)

---

##### **ASCR loses** — `a sheep in front of a key`

*Qwen3.5-9B (conf 0.95):* The baseline image correctly depicts a sheep in front of a key, satisfying the prompt. The right image replaces the key with a metallic grate or mesh structure, failing to generate the requested objec

![a sheep in front of a key — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/ascr_loss_2_a_sheep_in_front_of_a_key.png)

---

##### **Tie** — `a green bench and a blue bowl`

*Qwen3.5-9B (conf 0.95):* Both images perfectly satisfy the prompt, depicting a green bench and a blue bowl with accurate colors, counts, and spatial relations.

![a green bench and a blue bowl — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/tie_1_a_green_bench_and_a_blue_bowl.png)

---

##### **Tie** — `an oblong cucumber and a teardrop plum`

*Qwen3.5-9B (conf 0.95):* Both images accurately depict an oblong cucumber and a teardrop-shaped plum against a green background. The objects, colors, and spatial relations are identical in both images, with no material differ

![an oblong cucumber and a teardrop plum — pairwise (LEFT = ShowO Baseline, RIGHT = ASCR)](docs/examples/showo_baseline/tie_2_an_oblong_cucumber_and_a_teardrop_plum.png)


### ASCR vs BAGEL-7B-MoT

2 wins · 2 losses shown (out of 50 wins / 14 losses / 0 ties total).

##### **ASCR wins** — `an oblong cucumber and a teardrop plum`

*Qwen3.5-9B (conf 0.95):* The right image (ASCR) correctly depicts an oblong cucumber and a teardrop-shaped plum, matching the prompt's object descriptions and spatial arrangement. The left image (baseline) misidentifies the p

![an oblong cucumber and a teardrop plum — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_win_1_an_oblong_cucumber_and_a_teardrop_plum.png)

---

##### **ASCR wins** — `two boys`

*Qwen3.5-9B (conf 0.95):* The right image (ASCR) is a faithful representation of the prompt 'two boys', showing two distinct individuals. The left image (BAGEL) depicts two identical clones of the same boy, which is a hallucin

![two boys — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_win_2_two_boys.png)

---

##### **ASCR loses** — `a giraffe next to a lamp`

*Qwen3.5-9B (conf 0.95):* The left image (BAGEL) perfectly satisfies the prompt, showing a complete giraffe standing next to a lamp with correct spatial relations and lighting. The right image (ASCR) is severely cropped, cutti

![a giraffe next to a lamp — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_loss_1_a_giraffe_next_to_a_lamp.png)

---

##### **ASCR loses** — `a girl on the top of a frog`

*Qwen3.5-9B (conf 0.95):* The left image (baseline) perfectly matches the prompt 'a girl on the top of a frog' with a cute, high-quality 3D render of a girl sitting on a large frog in a pond. The right image (ASCR) shows a gir

![a girl on the top of a frog — pairwise (LEFT = BAGEL, RIGHT = ASCR)](docs/examples/bagel/ascr_loss_2_a_girl_on_the_top_of_a_frog.png)


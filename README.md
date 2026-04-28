# ASCR: Alternating Semantic-Confidence Revision

ASCR is a research prototype for studying and correcting confidence-semantic inconsistency in masked image-token generation. The central observation is that an image region can become confidence-stable during iterative denoising while still being semantically wrong with respect to the text prompt. Stage 1 starts with a zero-training implementation that uses a visible 4x4 grid and structured local semantic feedback to selectively reopen image-token regions instead of retrying the whole image.

This README is the project control document. It records the research plan, implementation plan, current progress, expected interfaces, cluster workflow, and GitHub synchronization policy. It should be updated whenever a meaningful implementation batch is completed.

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
- Localization interface: visible 4x4 grid over the decoded 256x256 image.
- Token grid target: 16x16 discrete image-token grid.
- Reopening rule: project selected 4x4 cells to 16x16 token regions and use fixed one-token dilation for Stage 1.
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

## Stage 1 System Overview

The practical Stage 1 loop is:

1. Receive the original prompt `P_orig`.
2. Run Show-o to an intermediate or completed image-token state `u`.
3. Decode `u` into an intermediate image `I_mid`.
4. Overlay a visible 4x4 grid to create `I_grid`.
5. Ask a local semantic evaluator to compare `P_orig` and `I_grid`.
6. Parse and validate structured semantic output `A_eval`.
7. Convert selected 4x4 cells into a 16x16 token reopening mask.
8. Apply fixed one-token dilation around selected token cells.
9. Compose a correction-conditioned prompt `P_cur`.
10. Reopen selected image-token regions and continue denoising.
11. Log every intermediate artifact and decision.
12. Stop when the evaluator returns no actionable semantic error, the iteration budget is exhausted, or fallback logic triggers abstention.

## Planned Repository Layout

The first implementation pass will build toward this structure:

```text
ASCR/
  README.md
  .gitignore
  pyproject.toml
  requirements/
    base.txt
    dev.txt
    local_vlm.txt
  configs/
    stage1_showo_local.yaml
    cluster_gpu.yaml
    cluster_gpu_shared.yaml
  ascr/
    __init__.py
    cli/
      run_stage1.py
      collect_traces.py
      validate_artifacts.py
    core/
      state.py
      schemas.py
      loop.py
      artifacts.py
    generators/
      base.py
      showo.py
      registry.py
    evaluators/
      base.py
      local_vlm.py
      schema_parser.py
    grids/
      overlay.py
      projection.py
    revision/
      selector.py
      remask.py
      prompt_composer.py
    benchmarks/
      prompts.py
      runner.py
      metrics.py
    traces/
      writer.py
      schema.py
    training/
      selector_model.py
      train_selector.py
      ddp.py
  jobs/
    stage1_debug_gpu_shared.sbatch
    stage1_run_gpu.sbatch
    stage2_train_selector_gpu.sbatch
  scripts/
    create_env.sh
    activate_env.sh
    run_stage1_debug.sh
    sync_github.sh
  docs/
    stage1_design.md
    benchmark_plan.md
    cluster_notes.md
  tests/
    test_schema_parser.py
    test_grid_projection.py
    test_prompt_composer.py
  data/
    README.md
  outputs/
  checkpoints/
  logs/
```

Some folders such as `outputs`, `checkpoints`, `logs`, model weights, and datasets are runtime artifacts and should not be committed.

## Stage 1 Implementation Plan

### S1.0 Repository Bootstrap

Status: in progress.

Tasks:

- Create project README with detailed roadmap and current status.
- Create `.gitignore` for Python, ML artifacts, Slurm outputs, local environments, and secrets.
- Initialize Git in `/grp01/cds_bdai/JianyuZhang/ASCR`.
- Connect remote `https://github.com/hmss2002/ASCR.git`.
- Commit and push confirmed updates to `main`.

### S1.1 Dedicated Environment

Status: planned.

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

Status: planned.

Tasks:

- Define `ASCRState` for prompt, token state, decoded image path, grid image path, evaluator output, masks, iteration counters, and artifact paths.
- Define `SemanticEvaluation` schema for structured local semantic feedback.
- Define `RegionSelection` schema for selected 4x4 grid cells, natural-language reason, confidence, and action type.
- Define `TokenReopenMask` schema for 16x16 reopening masks.
- Add strict parser and fallback behavior for malformed evaluator outputs.

Acceptance:

- Valid JSON parses into typed objects.
- Invalid JSON fails safely and triggers abstention.
- Unit tests cover valid, malformed, empty, and over-broad localization outputs.

### S1.3 Show-o Generator Adapter

Status: planned.

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

Status: planned.

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

Status: planned.

Tasks:

- Implement visible 4x4 grid overlay for 256x256 images.
- Add labels that match the evaluator coordinate vocabulary.
- Implement projection from selected 4x4 cells to 16x16 token cells.
- Add fixed one-token dilation for Stage 1.
- Keep image size and token grid size configurable.

Acceptance:

- Grid projection tests cover corners, edges, multiple cells, duplicate cells, and dilation boundaries.

### S1.6 ASCR Loop Orchestration

Status: planned.

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

Status: planned.

Tasks:

- Write trace records for each ASCR iteration.
- Include prompt, evaluator JSON, selected grid cells, projected token mask, correction prompt, and before-after artifact paths.
- Reserve optional fields for hidden states, confidence maps, revision gains, and human labels.

Acceptance:

- Stage 1 can produce a JSONL trace file suitable for future selector training.

### S1.8 Benchmark and Baselines

Status: planned.

Tasks:

- Create targeted benchmark prompt subsets for counting, spatial relations, color binding, negation, attribute binding, OCR, missing objects, and extra objects.
- Add baseline runners for whole-image retry, best-of-N reranking, verifier-only selection, generic inpainting adapter, confidence-only remask, semantic-only repair, and ASCR alternating.
- Add metrics for semantic improvement and collateral damage.

Acceptance:

- A small smoke benchmark can run locally with mock backends.
- Result tables are emitted as JSON and Markdown.

### S1.9 Cluster Jobs and Multi-GPU Readiness

Status: planned.

Tasks:

- Add Slurm scripts for `gpu_shared` debug runs.
- Add Slurm scripts for `gpu` longer runs.
- Support environment variables for partition, GPU count, CPU count, walltime, config path, and output directory.
- Keep Stage 1 inference compatible with single GPU.
- Reserve Stage 2 training scripts for `torchrun` and Slurm DDP.

Recommended usage:

- Use `gpu_shared` for short debugging, smoke tests, and interface validation.
- Use `gpu` for formal runs, long benchmark sweeps, and future multi-GPU training.

Acceptance:

- The same code path can run under interactive shell, `gpu_shared` Slurm job, or `gpu` Slurm job.

### S1.10 Documentation and Reproducibility

Status: planned.

Tasks:

- Maintain README after each confirmed update batch.
- Add design notes under `docs/`.
- Version prompt templates.
- Save config snapshots with every run.
- Record Git commit hash in each artifact directory.

Acceptance:

- A future reader can reproduce what was run from the artifact folder alone.

## Current Progress Log

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
- Configured GitHub remote `origin` for `https://github.com/hmss2002/ASCR.git`.

In progress:

- Push initial project-control files to GitHub `main`.

Next implementation batch:

1. Add `.gitignore`.
2. Initialize Git and remote.
3. Create dedicated `.venv` and environment scripts.
4. Add Python package skeleton.
5. Add schemas and grid projection tests.
6. Add mock Stage 1 loop before connecting real Show-o weights.

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
python -m ascr.cli.run_stage1 --help
python -m ascr.cli.run_stage1 --config configs/stage1_showo_local.yaml --dry-run
python -m pytest tests
```

The first Slurm milestones will be:

```bash
sbatch jobs/stage1_debug_gpu_shared.sbatch
sbatch jobs/stage1_run_gpu.sbatch
```

These commands are not expected to work until the corresponding implementation files are added.

## Stage 1 Acceptance Criteria

Stage 1 is considered complete when all of the following are true:

- A single prompt can run through the full ASCR loop with real Show-o integration.
- A dry-run mode can run without Show-o weights using mock adapters.
- The local semantic evaluator produces schema-validated JSON.
- Malformed evaluator output triggers safe fallback rather than remasking.
- 4x4 grid cells project correctly into 16x16 token masks.
- Fixed one-token dilation is implemented and tested.
- Each run saves decoded images, grid images, evaluator JSON, selected masks, prompts, configs, and trace JSONL.
- A small benchmark subset can compare ASCR with core baselines.
- Slurm scripts support both `gpu_shared` and `gpu`.
- README documents how to reproduce the latest working run.

## Open Decisions

These decisions are not blocking the repository bootstrap:

- Exact Show-o repository and checkpoint path.
- Exact local VLM/LLM backend for semantic evaluation.
- Final dataset storage path for large benchmarks.
- Whether generated paper figures should be tracked as lightweight examples or stored only as artifacts.

## Design Rule

Keep Stage 1 simple enough to prove the mechanism, but structure it so Stage 2 and Stage 3 do not require rewriting the project. The grid and JSON interface are implementation devices for the first prototype, not the final scientific claim.

# ASCR

ASCR (Alternating Semantic-Confidence Revision) is a research codebase for selective semantic reopening in masked image-token generation.

**Current mainline:** **Lumina-DiMOO + Qwen3.5-9B coarse selector** is now the recommended Stage-1 path. **Show-o** and **MMaDA-8B** remain in the repository as preserved comparison and legacy experiment lines.

This root `README.md` is now the **short front page**. The fully integrated long-form project
document is [`docs/ascr_master_guide.md`](docs/ascr_master_guide.md). The other README files are
only local notes for special directories:

- `data/README.md` - git-ignored datasets and generated payloads
- `external/README.md` - local third-party checkouts needed on the cluster

## Start here

| Need | Recommended entry |
| --- | --- |
| Integrated project guide | [`docs/ascr_master_guide.md`](docs/ascr_master_guide.md) |
| Current repo map | [`docs/architecture/repository_guide.md`](docs/architecture/repository_guide.md) |
| Stage-1 design principles | [`docs/architecture/stage1_design.md`](docs/architecture/stage1_design.md) |
| Key result artifacts | [`docs/results_overview.md`](docs/results_overview.md) |
| Dated experiment history | [`docs/history/experiment_changelog.md`](docs/history/experiment_changelog.md) |
| Archived full control document | [`docs/history/project_control_legacy.md`](docs/history/project_control_legacy.md) |

## Current recommended workflow

For the mainline Lumina-DiMOO Stage-1 pipeline, start from these files:

| Role | File |
| --- | --- |
| Primary config | `configs/stage1_lumina_qwen9b_coarse_hq.yaml` |
| Batch job | `jobs/stage1_lumina_qwen_coarse_hard64_8gpu.sbatch` |
| Run script | `scripts/run_lumina_qwen_coarse_hard64.py` |
| Generator implementation | `ascr/generators/lumina_dimoo.py`, `ascr/generators/lumina_native.py` |
| Core revision loop | `ascr/core/loop.py` |
| Evaluator registry | `ascr/evaluators/registry.py` |

## Model status

| Status | Model family | Notes |
| --- | --- | --- |
| **Primary** | Lumina-DiMOO | Mainline discrete-diffusion base model going forward |
| Comparison | Show-o | Original Stage-1 baseline and direct-token experiments |
| Comparison | MMaDA-8B | Self-eval and Qwen-selector transfer experiments |
| External reference | BAGEL-7B | Benchmark comparison only, not ASCR mainline |

## Repository layout

The code is already modular under `ascr/`, but the operational assets grew historically around multiple experiments. The recommended way to browse the project is:

- `ascr/` - core package: generators, evaluators, loop, selectors, CLI
- `configs/` - experiment configs; still flat for compatibility, but see the repository guide for the logical grouping
- `jobs/` - Slurm entrypoints
- `scripts/` - orchestration, judging, data prep, download helpers
- `docs/` - architecture notes, results, examples, and historical records
- `tests/` - mock-heavy regression coverage for wiring and loop behavior

## Notes on the cleanup

- No historical experiment assets were deleted.
- Long-form narrative, blueprint framing, and workflow material are now integrated in `docs/ascr_master_guide.md`.
- Historical material remains preserved in the archive and changelog docs.
- The next cleanup target, if needed, is the flat `configs/` / `jobs/` / `scripts/` layout; for now the repository guide provides the stable navigation layer without breaking existing paths.

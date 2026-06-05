# Repository Guide

This document is the practical map of the current repository.

## 1. Mainline vs preserved experiment lines

| Track | Purpose | Main files |
| --- | --- | --- |
| **Lumina-DiMOO mainline** | Current recommended Stage-1 path | `configs/stage1_lumina_qwen9b_coarse_hq.yaml`, `scripts/run_lumina_qwen_coarse_hard64.py`, `jobs/stage1_lumina_qwen_coarse_hard64_8gpu.sbatch`, `ascr/generators/lumina_dimoo.py` |
| Show-o preserved line | Original ASCR baseline, direct-token experiments, legacy comparison path | `configs/stage1_showo_qwen35_9b*.yaml`, `scripts/run_showo_qwen_coarse_hard64.py`, `ascr/generators/showo.py`, `ascr/generators/showo_native.py` |
| MMaDA-8B preserved line | Self-eval and transferred selector experiments | `configs/stage1_mmada8b_*.yaml`, `scripts/run_mmada_*`, `ascr/generators/mmada.py`, `ascr/generators/mmada_native.py`, `ascr/evaluators/mmada_self*.py` |

## 2. Top-level directories

| Path | What it contains |
| --- | --- |
| `ascr/` | Core Python package |
| `configs/` | Experiment YAMLs, benchmark metadata, cluster templates |
| `jobs/` | Slurm job entrypoints |
| `scripts/` | Local runners, judges, download/setup helpers, benchmark utilities |
| `docs/` | Architecture notes, results, examples, and history |
| `tests/` | Wiring, loop, selector, and evaluator regression tests |
| `data/` | Git-ignored datasets and generated payloads |
| `external/` | Local third-party source trees used on cluster machines |

## 3. `ascr/` package map

| Subpackage | Role |
| --- | --- |
| `ascr/core/` | Config loading, loop state, artifacts, schemas |
| `ascr/generators/` | Show-o, MMaDA, Lumina generators and native engines |
| `ascr/evaluators/` | Qwen-VL, MMaDA self-eval, remote eval, mock and legacy evaluators |
| `ascr/revision/` | Semantic-to-mask selectors and prompt composition |
| `ascr/grids/` | Overlay rendering and coarse-to-token projection helpers |
| `ascr/cli/` | Single-run and comparison CLIs |
| `ascr/benchmarks/` | Benchmark helpers and markdown/json summaries |
| `ascr/training/` | Stage-2 training placeholders and DDP helpers |

## 4. How to browse the flat operational assets

The repository still keeps `configs/`, `jobs/`, and `scripts/` largely flat to avoid breaking
existing Slurm commands and historical notebooks. Use the prefixes below as the logical grouping:

| Prefix / pattern | Meaning |
| --- | --- |
| `stage1_lumina_*` | Lumina-DiMOO mainline |
| `stage1_showo_*` | Show-o Stage-1 runs and direct-token variants |
| `stage1_mmada_*` | MMaDA-8B runs |
| `judge_*` | External judging scripts |
| `download_*` | Model/bootstrap helpers |
| `prepare_*` / `build_*` / `merge_*` / `summarize_*` | Benchmark and artifact utilities |

## 5. Recommended entrypoints

### If you want the current mainline

1. Read `configs/stage1_lumina_qwen9b_coarse_hq.yaml`.
2. Run `scripts/run_lumina_qwen_coarse_hard64.py` or the matching Slurm job.
3. Inspect generator wiring in `ascr/generators/lumina_dimoo.py`.

### If you want the original ASCR baseline line

1. Start with `configs/stage1_showo_qwen35_9b.yaml` or `configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml`.
2. Read `ascr/generators/showo_native.py` and `ascr/evaluators/qwen_vl.py`.

### If you want the transfer/self-eval experiments

1. Start with `configs/stage1_mmada8b_qwen9b_coarse.yaml` or `configs/stage1_mmada8b_self_coarse.yaml`.
2. Read `ascr/generators/mmada_native.py` plus `ascr/evaluators/mmada_self*.py`.

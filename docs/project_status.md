# Project Status

Last updated: 2026-05-14.

## Completed

- Qwen3.5-9B downloaded from `Qwen/Qwen3.5-9B` and verified locally at `models/qwen3.5-9b`.
- Single-GPU Stage 1 full-flow smoke completed as Slurm job `68379`.
- 8-GPU one-worker-per-GPU Stage 1 smoke completed as Slurm job `68386`.
- Qwen3.5-9B is now the default evaluator in the main config, CLI defaults, registry fallback, and run scripts.
- Historical Qwen3.6/AWQ experiment files are grouped under `configs/experiments/qwen36/` and `jobs/experiments/qwen36/`.
- DrawBench public prompts are prepared as `configs/prompts/drawbench_smoke8.txt` and `configs/prompts/drawbench_all.txt`.
- Raw and processed benchmark payload folders are ignored by git.

## Ready To Run

- `bash scripts/download_qwen35_9b_snapshot.sh` downloads or refreshes the default evaluator snapshot.
- `bash scripts/run_stage1_showo_compare.sh` runs the default single-process comparison path.
- `bash scripts/run_stage1_showo_compare_parallel.sh` runs the default parallel comparison path.
- `python scripts/prepare_drawbench_prompts.py --smoke-limit 8` prepares DrawBench prompt files.
- `sbatch jobs/stage1_drawbench_qwen35_9b_smoke8.sbatch` runs the public DrawBench smoke subset.

## Not Yet Done

- DrawBench smoke results need to be generated and summarized.
- A final independent judge protocol is still needed before making public benchmark claims.
- Full DrawBench and T2I-CompBench sweeps have not been run.
- T2I-CompBench import needs parquet dependencies such as `pandas` plus `pyarrow` or `fastparquet`.
- README and benchmark docs should be kept updated after each run with job ids and exact counts.

## Interpretation Notes

- `configs/prompts/stage1_complex_prompts.txt` is an internal development smoke suite, not a public benchmark.
- Current `comparison.verdict` values come from a small heuristic metric and should not be presented as a fair benchmark result.
- Qwen3.5-9B is reliable enough for the ASCR repair loop on this cluster, but using the same model as both repair evaluator and final judge should be disclosed.

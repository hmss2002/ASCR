# Project Status

Last updated: 2026-05-14.

## Completed

- Qwen3.5-9B downloaded from `Qwen/Qwen3.5-9B` and verified locally at `models/qwen3.5-9b`.
- Single-GPU Stage 1 full-flow smoke completed as Slurm job `68379`.
- 8-GPU one-worker-per-GPU Stage 1 smoke completed as Slurm job `68386`.
- Qwen3.5-9B is now the default evaluator in the main config, CLI defaults, registry fallback, and run scripts.
- Historical Qwen3.6/AWQ experiment files are grouped under `configs/experiments/qwen36/` and `jobs/experiments/qwen36/`.
- DrawBench public prompts are prepared as `configs/prompts/drawbench_smoke8.txt` and `configs/prompts/drawbench_all.txt`.
- T2I-CompBench prompts are prepared as `configs/prompts/t2i_compbench_hard_smoke8.txt` and `configs/prompts/t2i_compbench_hard64.txt`.
- Clean final-image paired judging is implemented in `scripts/judge_showo_ascr_pairs_qwen.py`.
- Single-process comparison supports `--reuse-models`; the shell wrapper enables it with `REUSE_MODELS=1`.
- Raw and processed benchmark payload folders are ignored by git.

## Ready To Run

- `bash scripts/download_qwen35_9b_snapshot.sh` downloads or refreshes the default evaluator snapshot.
- `bash scripts/run_stage1_showo_compare.sh` runs the default single-process comparison path.
- `bash scripts/run_stage1_showo_compare_parallel.sh` runs the default parallel comparison path.
- `python scripts/prepare_drawbench_prompts.py --smoke-limit 8` prepares DrawBench prompt files.
- `python scripts/prepare_t2i_compbench_prompts.py` prepares T2I-CompBench prompt files.
- `sbatch jobs/stage1_drawbench_qwen35_9b_smoke8.sbatch` runs the public DrawBench smoke subset.
- `REUSE_MODELS=1 PROMPT_LIMIT=2 sbatch jobs/stage1_t2i_compbench_qwen35_9b_smoke1.sbatch` runs a sequential T2I reuse smoke.
- `sbatch jobs/stage1_t2i_compbench_qwen35_9b_smoke8.sbatch` runs the 8-GPU T2I smoke subset.

## Not Yet Done

- DrawBench 8-prompt smoke remains pending resources.
- A final judge independent of the ASCR repair evaluator is still needed before making public benchmark claims.
- Full DrawBench and T2I-CompBench sweeps have not been run.
- T2I-CompBench hard64 has not been run yet.
- README and benchmark docs should be kept updated after each run with job ids and exact counts.

## Interpretation Notes

- `configs/prompts/stage1_complex_prompts.txt` is an internal development smoke suite, not a public benchmark.
- Current `comparison.verdict` values come from a small heuristic metric and should not be presented as a fair benchmark result.
- Qwen3.5-9B is reliable enough for the ASCR repair loop on this cluster, but using the same model as both repair evaluator and final judge should be disclosed.

## DrawBench Smoke Results

- 2026-05-14 job `68440`: 1 DrawBench prompt on 1 GPU, `COMPLETED 0:0` in `00:00:56`.
- Prompt: `A red colored car.`
- Output: `outputs/benchmarks_drawbench_qwen35_smoke1gpu/showo_ascr-20260514-034344/comparison.json`.
- Counts: 1 comparison, 1 evaluator JSON, 0 parser errors, 0 abstains.
- ASCR summary: `stop_reason: no_semantic_error`, `evaluator_calls: 1`, `ascr_insertions: 0`.
- Heuristic comparison: baseline 1.0, ASCR 1.0, verdict `tie_or_unclear`.
- 2026-05-14 job `68439`: 8 DrawBench prompts on 8 GPUs, submitted and still pending resources.

This confirms the public DrawBench prompt path runs end to end. It does not yet establish benchmark superiority because the current comparison verdict remains heuristic.

## T2I-CompBench Smoke Results

- 2026-05-14 job `68441`: 1 T2I-CompBench prompt on 1 GPU, completed.
- 2026-05-14 job `68443`: 8 T2I-CompBench prompts on 1 GPU fallback, completed and wrote `outputs/benchmarks_t2i_compbench_qwen35_smoke8_1gpu/showo_ascr-20260514-040615/suite.json`.
- Heuristic suite summary for job `68443`: `ascr_improved=1`, `ascr_regressed=1`, `tie_or_unclear=6`, `total_evaluator_calls=14`, `total_ascr_insertions=6`.
- 2026-05-14 job `68444`: clean final-image Qwen judge over the 8-prompt suite, `COMPLETED 0:0` in `00:00:50`.
- Clean final-image judge counts for job `68444`: `baseline_pass=8`, `ascr_pass=8`, `both_pass=8`.
- 2026-05-14 job `68445`: 2-prompt `REUSE_MODELS=1` validation, `COMPLETED 0:0` in `00:02:21`; suite verdicts were `ascr_improved=1`, `tie_or_unclear=1`, with `total_evaluator_calls=3` and `total_ascr_insertions=1`.
- Clean final-image judge counts for job `68445`: `baseline_pass=2`, `ascr_pass=2`, `both_pass=2`.
- 2026-05-14 job `68442`: 8-GPU T2I smoke submitted and pending priority/resources.

The smoke8 set confirms that the harder public prompt path, suite aggregation, reuse path, and clean final-image judge work. It does not establish ASCR superiority because baseline and ASCR both pass every clean-final prompt under this judge.

## Runtime Cleanup Results

- `--reuse-models` reuses the baseline generator, ASCR generator, and Qwen evaluator across prompts in the single-process comparison path.
- Compatible baseline and ASCR `ShowOAdapter` instances share one underlying native Show-o engine, avoiding a second Show-o weight load in sequential runs.
- `jobs/stage1_t2i_compbench_qwen35_9b_smoke1.sbatch` now defaults `REUSE_MODELS=1` and judges the latest `suite.json` when running multiple prompts, falling back to `comparison.json` for single-prompt outputs.

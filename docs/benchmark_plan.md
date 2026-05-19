# Benchmark Plan

Stage 1 benchmarks should answer a narrower question than generic image quality: does ASCR improve prompt following over the same Show-o baseline image state after semantic-confidence repair?

## Current Baseline State

- Generator: local Show-o native token loop.
- Default evaluator inside the ASCR loop: `Qwen/Qwen3.5-9B`, stored locally at `models/qwen3.5-9b`.
- Default config: `configs/stage1_showo_qwen35_9b.yaml`.
- Development smoke prompts: `configs/prompts/stage1_complex_prompts.txt`.
- Public prompt suites now prepared: DrawBench and T2I-CompBench smoke/hard subsets.

The original heuristic metric in `ascr/benchmarks/metrics.py` is intentionally small. It can catch simple color and spatial regressions, but it is not a fair public benchmark judge. Treat `comparison.verdict` as a smoke-test signal. Use the clean final-image judge for automated prompt-following checks, and disclose that Qwen is not independent when reused as both repair evaluator and final judge.

## Verified Qwen3.5-9B Feasibility

| Scope | Job | Result | Evidence |
| --- | --- | --- | --- |
| Single GPU full-flow smoke | `68379` | completed in `00:02:43` | wrote `comparison.json` |
| 8-GPU parallel smoke | `68386` | completed in `00:07:32` | 8 comparisons, 29 evaluator JSON files, 0 parser errors, 0 abstains |

This means Qwen3.5-9B is practical as the default Stage 1 evaluator on the current cluster, including the one-worker-per-GPU parallel path.

## DrawBench

DrawBench is the first public prompt-only suite wired into this repository. It provides prompts but not reference images, so the benchmark protocol must generate paired baseline/ASCR images and then judge prompt following.

Prepared files:

- `scripts/prepare_drawbench_prompts.py`: downloads or reuses the public `sayakpaul/drawbench` CSV and writes ASCR prompt text files.
- `configs/prompts/drawbench_smoke8.txt`: 8-prompt smoke subset across major categories.
- `configs/prompts/drawbench_all.txt`: all 200 DrawBench prompts.
- `jobs/stage1_drawbench_qwen35_9b_smoke8.sbatch`: 8-GPU smoke job.

Prepare prompts:

```bash
python scripts/prepare_drawbench_prompts.py --smoke-limit 8
```

Run the 8-prompt smoke:

```bash
sbatch jobs/stage1_drawbench_qwen35_9b_smoke8.sbatch
```

Scale to the full suite after smoke succeeds:

```bash
PROMPTS_FILE=configs/prompts/drawbench_all.txt PROMPT_LIMIT=200 REPEAT_COUNT=1 sbatch jobs/stage1_qwen35_9b_parallel8.sbatch
```

## Fair Scoring Protocol

Minimum credible protocol:

1. Generate a baseline image and ASCR image from the same initial Show-o state for each prompt and seed.
2. Run a judge that has not been used to decide the final comparison verdict during generation, or at least report that Qwen is both the repair evaluator and judge if reusing it.
3. Score baseline and ASCR independently for prompt following.
4. Report win/tie/loss, parser failures, abstains, and per-category breakdowns.
5. Keep generated media under `outputs/` and commit only configs, scripts, and summarized metrics.

Candidate judges:

- Qwen3.5-9B final judge: easiest immediate path, consistent with the repair evaluator, but must disclose that it is not independent if reused.
- TIFA/VQA-style judge: stronger semantic question answering, requires extra implementation and model dependencies.
- VQAScore or CLIP-derived metrics: useful as auxiliary signals, not sufficient alone for spatial/counting/OCR claims.
- GenEval-style object checks: stronger for object/count/spatial categories, heavier setup.
- Human/audited subset: best sanity check for a small sample, especially DrawBench text and rare-word cases.

## Other Public Benchmarks

- T2I-CompBench is wired through `scripts/prepare_t2i_compbench_prompts.py`; the parquet path works in `.venv-qwen36` with `pyarrow==24.0.0`.
- GenEval and TIFA are better aligned with prompt-following claims, but require additional code and model downloads.
- The repository should not claim ASCR improves DrawBench/T2I-CompBench until the judge protocol above is implemented and run beyond the smoke subset.

## Reporting Template

Each benchmark run should report:

- config path and git commit.
- model checkpoint path for Show-o and Qwen.
- prompt suite, prompt count, repeat count, and seed policy.
- number of successful comparisons and failed runs.
- evaluator parser errors and abstains.
- baseline wins, ASCR wins, ties, and category breakdown.
- representative image grid stored as an artifact, not committed unless intentionally small.

## T2I-CompBench Smoke Protocol

Prepared files:

- `scripts/prepare_t2i_compbench_prompts.py`: exports prompts from `NinaKarine/t2i-compbench`.
- `configs/prompts/t2i_compbench_hard_smoke8.txt`: 8 unique smoke prompts across harder categories.
- `configs/prompts/t2i_compbench_hard64.txt`: 64-prompt follow-up subset.
- `jobs/stage1_t2i_compbench_qwen35_9b_smoke1.sbatch`: 1-GPU smoke/sequential runner with `REUSE_MODELS=1` by default.
- `scripts/judge_showo_ascr_pairs_qwen.py`: clean final-image paired judge.

Run a small sequential smoke:

```bash
PROMPT_LIMIT=2 REUSE_MODELS=1 sbatch jobs/stage1_t2i_compbench_qwen35_9b_smoke1.sbatch
```

Run the default hard64 checkpoint:

```bash
sbatch jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch
```

Important scoring rule: ASCR grid images are only localization diagnostics. Final scoring uses clean baseline and ASCR final images.

Latest hard64 interpretation: job `68660` completed 64 prompts with 8 model-reuse shards inside one 8-GPU Slurm allocation. Pairwise Qwen judge found `ascr_win=13`, `ascr_loss=6`, `pairwise_tie=45` for net `+7`. Clean final-image Qwen pass/fail found `ascr_pass=57/64` and `baseline_pass=53/64` for net `+4`. This is the Stage 1 Phase 1 checkpoint; future config exploration should report deltas against it.

## Stage 1 Phase 1 Default Settings

Use these settings as the default anchor for follow-up experiments:

- Command: sbatch jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch.
- Slurm: one job with #SBATCH --gres=gpu:8.
- Workers: SHARD_WORKERS=8, one CUDA_VISIBLE_DEVICES shard per worker.
- Prompt suite: configs/prompts/t2i_compbench_hard64.txt, PROMPT_LIMIT=64.
- Runtime: REUSE_MODELS=1.
- ASCR start: ASCR_START_MODE=baseline.
- Generation: GENERATION_TIMESTEPS=18, GUIDANCE_SCALE=4, MAX_ITERATIONS=8, REPEAT_COUNT=1, SEED_STEP=1.
- Judges: report both Qwen clean pass/fail and Qwen side-by-side pairwise counts.
- Current checkpoint: job 68660, pairwise net +7 and clean pass net +4 on hard64.

# Stage 1 Phase 1 Summary - 2026-05-19

This is the current Stage 1 baseline-vs-ASCR checkpoint. The default evidence run is T2I-CompBench hard64 with Qwen3.5-9B, same-initial-state baseline mode, one Slurm allocation requesting 8 GPUs, and eight model-reuse workers inside that allocation.

## Default Command

```bash
sbatch jobs/stage1_t2i_compbench_qwen35_9b_hard64_8gpu_reuse.sbatch
```

## Default Runtime Shape

- Slurm job: one job requesting #SBATCH --gres=gpu:8.
- Verified allocation from the completed run: billing=32,cpu=32,gres/gpu=8,mem=192G,node=1.
- Internal workers: SHARD_WORKERS=8.
- Prompt split: contiguous shards from configs/prompts/t2i_compbench_hard64.txt, 8 prompts per worker for the default 64-prompt run.
- Per-worker GPU binding: CUDA_VISIBLE_DEVICES is set to the worker index.
- Model reuse: REUSE_MODELS=1, so each worker keeps the baseline generator, ASCR generator, and Qwen evaluator loaded across its shard.
- Runner: scripts/run_stage1_showo_compare_sharded_reuse.sh.

## Default Generation and ASCR Settings

- Config: configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml.
- Prompt file: configs/prompts/t2i_compbench_hard64.txt.
- Prompt limit: 64.
- ASCR start mode: baseline.
- Max iterations: 8.
- Generation timesteps: 18.
- Guidance scale: 4.
- Repeat count: 1.
- Seed step: 1.
- Safety fallback: return_initial_on_max_error: true in the Qwen3.5 configs.
- Qwen model path: models/qwen3.5-9b, offline/local-files mode enabled by default in the Slurm job.

## Completed Evidence Run

- Job: 68660.
- Status: COMPLETED, exit code 0:0.
- Elapsed: 00:16:38.
- Run root: outputs/benchmarks_t2i_compbench_qwen35_hard64_slurm8gpu_reuse_20260519_191652.
- Prompt count: 64.
- Completed shards: 8.

## Result Summary

Qwen side-by-side pairwise judge:

| Metric | Count |
| --- | ---: |
| ASCR win | 13 |
| ASCR loss | 6 |
| Tie | 45 |
| Net ASCR wins | +7 |

Qwen clean final-image pass/fail judge:

| Metric | Count |
| --- | ---: |
| ASCR pass | 57 / 64 |
| Baseline pass | 53 / 64 |
| Both pass | 53 |
| Both fail | 7 |
| Net ASCR pass gain | +4 |

## Interpretation

This is a reasonable first Stage 1 checkpoint: both automated judge views favor ASCR over the native Show-o baseline on the hard64 subset. The pairwise judge measures direct preference between baseline and ASCR images; the clean pass/fail judge measures whether each image independently satisfies the prompt. Both judges use Qwen3.5-9B, which is also the ASCR loop evaluator, so these numbers are benchmark signals rather than independent human evidence.

## Next Exploration Tracks

- Sweep ASCR thresholds and max_iterations around this default.
- Try partial start mode only as an explicit experiment, not as the default result path.
- Add a more independent judge or official T2I-CompBench metrics on the same clean image outputs.
- Run category-level breakdowns once enough prompts are available.

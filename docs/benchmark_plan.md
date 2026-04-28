# Benchmark Plan

Stage 1 benchmarks should target confidence-semantic inconsistency rather than generic image quality. Initial subsets include counting, spatial relations, color binding, negation, attribute binding, OCR, missing objects, and extra objects.

Baselines should include whole-image retry, best-of-N reranking, verifier-only selection, generic inpainting adapters, confidence-only remask, semantic-only repair, and the full ASCR alternating loop.

## Current Runnable Comparison

The current local comparison path is:

```bash
source .venv/bin/activate
bash scripts/run_stage1_showo_compare.sh
```

It generates one original Show-o baseline image, then starts ASCR from that same baseline image state. This avoids comparing two independent random samples.

Current local evaluator status:

- Backend: heuristic image evaluator.
- Supported smoke checks: color presence and red-left-of-blue spatial relation.
- Not yet suitable for paper-quality claims.

Latest fair single-prompt results for `A red cube left of a blue sphere`:

| Setting | Baseline | ASCR | Verdict |
| --- | ---: | ---: | --- |
| 4 steps | 0.992772 | 0.992772 | tie_or_unclear |
| 18 steps | 0.874457 | 0.874457 | tie_or_unclear |

Interpretation: the Stage 1 code path is runnable, logs artifacts, and compares fairly, but the current heuristic single-prompt evidence does not show improvement over original Show-o. The next benchmark milestone is a VLM-backed evaluator and a multi-prompt, multi-seed sweep.

## Slurm Entry Points

Use `gpu_shared` for smoke/debug:

```bash
sbatch jobs/stage1_compare_gpu_shared.sbatch
```

Use `gpu` for longer comparison runs:

```bash
sbatch jobs/stage1_compare_gpu.sbatch
```

Runtime knobs are exposed through environment variables in `scripts/run_stage1_showo_compare.sh`, including `PROMPT`, `CONFIG`, `OUTPUT_DIR`, `GENERATION_TIMESTEPS`, `GUIDANCE_SCALE`, and `MAX_ITERATIONS`.

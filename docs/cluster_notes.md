# Cluster Notes

Use gpu_shared for smoke tests, dry runs, parser debugging, and quick artifact checks. Use gpu for longer runs, benchmark sweeps, and future multi-GPU training.

The same Python entry point should run from an interactive shell, a gpu_shared job, or a gpu job. Stage 2 training reserves torchrun and Slurm DDP entry points.

## Stage 1 Compare Jobs

Smoke/debug path:

```bash
sbatch jobs/stage1_compare_gpu_shared.sbatch
```

Longer single-GPU path:

```bash
sbatch jobs/stage1_compare_gpu.sbatch
```

The compare script accepts `PROMPT`, `CONFIG`, `OUTPUT_DIR`, `GENERATION_TIMESTEPS`, `GUIDANCE_SCALE`, and `MAX_ITERATIONS` through the environment. Single-image Show-o inference is single-GPU; use multi-GPU for prompt sweeps or Stage 2 training.

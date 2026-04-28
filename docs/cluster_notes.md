# Cluster Notes

Use gpu_shared for smoke tests, dry runs, parser debugging, and quick artifact checks. Use gpu for longer runs, benchmark sweeps, and future multi-GPU training.

The same Python entry point should run from an interactive shell, a gpu_shared job, or a gpu job. Stage 2 training reserves torchrun and Slurm DDP entry points.

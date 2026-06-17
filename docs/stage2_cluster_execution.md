# Stage 2 cluster execution

## Goal

Use school GPU capacity primarily for **parallel Stage 2 data generation and ablations**, not for fragile cross-node communication.

The target setup is up to **56 GPUs total**, split as:

- **Cluster A:** up to 28 GPUs
- **Cluster B:** up to 28 GPUs

## Recommended execution model

### 1. Teacher-only data generation

Treat each prompt or prompt chunk as an independent job.

- No DDP required.
- No cross-node communication required.
- Each shard writes to its own output directory.
- Merge later from `trace.jsonl`.

Preferred scripts:

- `jobs/stage2/lumina/stage2_lumina_qwen37_teacher_single_gpu_array.sbatch`
- `jobs/stage2/lumina/stage2_lumina_qwen37_teacher_8gpu_node.sbatch`

### 2. Dataset merge

After shard generation completes:

```bash
python -m ascr.training.build_stage2_dataset \
  outputs/stage2_lumina_qwen37_teacher_hq \
  --output outputs/stage2_lumina_qwen37_teacher_hq/stage2_teacher_dataset.jsonl \
  --skipped-report outputs/stage2_lumina_qwen37_teacher_hq/stage2_teacher_skipped.jsonl
```

### 3. Replay selector training

Replay baselines do not need heavy GPU allocation.

```bash
python -m ascr.training.train_selector \
  --dataset outputs/stage2_lumina_qwen37_teacher_hq/stage2_teacher_dataset.jsonl \
  --output-dir checkpoints/stage2_selector_replay
```

### 4. Learned selector training

The repo now includes a lightweight learned coarse-selector baseline. Start with **single-node 8 GPU** training if torch/GPU is available, or CPU for smoke runs.

```bash
torchrun --standalone --nproc_per_node=8 -m ascr.training.train_selector \
  --dataset outputs/stage2_lumina_qwen37_teacher_hq/stage2_teacher_dataset.jsonl \
  --output-dir checkpoints/stage2_selector_replay \
  --mode learned_coarse
```

For two clusters, prefer running different seeds or ablations independently rather than forcing cross-cluster distributed training.

## Sharding strategy

### Teacher-only ASCR sharding

- `PROMPT_FILE` contains one prompt per line.
- `SLURM_ARRAY_TASK_ID` identifies the shard.
- `PROMPTS_PER_TASK` controls one-prompt vs chunk-per-task behavior.
- Each shard writes under:

```text
outputs/stage2_lumina_qwen37_teacher_hq/shard_<task_id>/
```

### Resume strategy

- Skip shard directories that already contain a finished trace or summary.
- Resubmit only failed array indices.
- Keep output roots stable across retries.

## API throttling

Separate these two controls:

1. **Job-array concurrency**: how many prompt shards run at once.
2. **Per-job API concurrency**: how many teacher API requests one worker is allowed to issue internally.

Current Stage 2 generation is effectively serial per prompt, so start with:

- `api_concurrency: 1`
- `api_retry: 3`
- `api_timeout: 120`
- `api_backoff: 2.0`

If the teacher endpoint rate-limits, reduce array concurrency first.

## Recommended run scales

### Minimal smoke run

- 1 GPU
- 1 prompt
- single `run_stage1` invocation with the Stage 2 config

### Medium run

- 8 GPUs on one node
- 8 independent workers
- 1 prompt or small chunk per worker

### Full run

- up to 56 concurrent single-GPU workers
- 28 on each cluster
- two independent submissions writing to disjoint cluster output roots

## Ablation plan

Recommended ablations:

1. Lumina baseline
2. Lumina + Stage 1 Qwen coarse selector
3. Lumina + Stage 2 teacher-only selector
4. Replay selector checkpoint
5. Learned selector checkpoint

## What to record

For each shard or training run, collect:

- prompt count
- successful traces
- skipped / failed traces
- mean teacher latency
- API usage totals if returned by provider
- selected cell count distribution
- clean pass rate
- pairwise regressions

The point of the Stage 2 cluster workflow is reproducible throughput, not nominal GPU utilization.

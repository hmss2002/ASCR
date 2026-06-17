# Two-cluster Stage 2 teacher execution plan

## Prompt splitting

Split the prompt list into two disjoint files before submission:

- `prompts_cluster_a.txt`
- `prompts_cluster_b.txt`

Recommended split: contiguous halves for easier resume and accounting.

## Recommended submissions for 28 GPU per cluster

### Strategy A: 1 GPU / job (recommended default)

On each cluster:

```bash
sbatch --array=0-27%28 \
  --export=ALL,PROMPT_FILE=prompts_cluster_a.txt,PROMPTS_PER_TASK=1 \
  jobs/stage2/lumina/stage2_lumina_qwen37_teacher_single_gpu_array.sbatch
```

Repeat on cluster B with `prompts_cluster_b.txt`.

Advantages:

- maximum shard independence
- easiest resume
- best fit if Lumina effectively uses one GPU per process
- no distributed failure modes

### Strategy B: 8 GPU / node

On each cluster, submit up to 3 full 8-GPU nodes plus one partial node if available:

```bash
sbatch --export=ALL,PROMPT_FILE=prompts_cluster_a.txt,NODE_INDEX=0,NODE_COUNT=4 \
  jobs/stage2/lumina/stage2_lumina_qwen37_teacher_8gpu_node.sbatch
```

Repeat with `NODE_INDEX=1,2,3`.

Advantages:

- fewer scheduler objects
- easier per-node log collection

Trade-off:

- coarser resume granularity than 1 GPU / job

## Avoiding output collisions

Use disjoint output roots:

- `outputs/stage2_lumina_qwen37_teacher_hq/cluster_a/...`
- `outputs/stage2_lumina_qwen37_teacher_hq/cluster_b/...`

Never let both clusters write into the same shard directory.

## Merging traces

After both clusters finish:

```bash
python -m ascr.training.build_stage2_dataset \
  outputs/stage2_lumina_qwen37_teacher_hq/cluster_a \
  outputs/stage2_lumina_qwen37_teacher_hq/cluster_b \
  --output outputs/stage2_lumina_qwen37_teacher_hq/stage2_teacher_dataset.jsonl \
  --skipped-report outputs/stage2_lumina_qwen37_teacher_hq/stage2_teacher_skipped.jsonl
```

## Resuming failed shards

- resubmit only missing array indices for Strategy A
- resubmit only failed node jobs for Strategy B
- keep prompt files unchanged across retries
- keep shard directory names stable

## API usage and latency accounting

For each cluster, aggregate:

- number of completed traces
- number of abstentions / failures
- total API calls
- total usage tokens if returned by provider
- mean latency per diagnosis

Recommended practice:

- write cluster-local summaries first
- merge cluster summaries only after raw shard completion is stable

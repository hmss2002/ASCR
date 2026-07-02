# H200 Fast Path

This checkout now treats the current H200 server as the primary execution
target. L40S profiles remain available only as explicit fallbacks.

## Stage 3 token repair

Use the H200 array mode:

```bash
MODE=submit_clean_h200 bash scripts/training/run_stage3_token_repair_dataset.sh
```

The default H200 split is 32 short array tasks with 4 GPUs each. On the current
server, Slurm is allowing 16 GPUs concurrently for this account, so the array
keeps the full allowance busy while completed shards hand off to pending shards.

The clean-token generator also exposes Lumina sampling knobs for controlled
speed/quality experiments:

```bash
GENERATION_TIMESTEPS=32 MODE=submit_clean_h200 \
  bash scripts/training/run_stage3_token_repair_dataset.sh
```

The default remains `GENERATION_TIMESTEPS=64` for quality parity. The chosen
timesteps, guidance scale, and temperature are recorded in each clean-token
manifest row.

Clean-token generation is resumable at the shard level: existing token files
with the expected grid length are reused without loading the Lumina engine.
Invalid or incomplete token files are regenerated.
Each shard output directory also takes an exclusive lock while writing, so a
resumed job and an opportunistic tail job cannot write the same manifest at the
same time.

If the tail of the 4-GPU array is stuck because only one or two account GPUs
become available at a time, split selected nodes into 1-GPU shards:

```bash
CLEAN_OUTPUT_ROOT=outputs/stage3_token_repair/clean_tokens_h200_32x4 \
CLEAN_SHARD_NODE_START=24 \
CLEAN_SHARD_NODE_END=31 \
CLEAN_SHARD_ARRAY_LIMIT=16 \
MODE=submit_clean_h200_shards \
  bash scripts/training/run_stage3_token_repair_dataset.sh
```

This writes the same `node_XXXX/gpu_YY` shard directories as the normal H200
array, so downstream merge/report/build commands do not change.

To submit the whole token-repair pipeline through Stage 4:

```bash
bash scripts/training/run_h200_token_repair_pipeline.sh
```

The pipeline submits Stage 3 clean-token generation, then uses an `afterok`
dependency to merge clean manifests, validate the merged manifest, build the
40k repair dataset, prepare SFT data, submit Stage 4 LoRA training, write a
speed report, and queue the LoRA probe.

To validate clean-token outputs manually after merge:

```bash
MODE=merge_clean bash scripts/training/run_stage3_token_repair_dataset.sh
MODE=report_clean REPORT_MIN_ROWS=10000 \
  bash scripts/training/run_stage3_token_repair_dataset.sh
```

The report is written under the clean-token output root and checks row count,
duplicate sample IDs, duplicate prompt indexes, missing referenced token files,
unmanifested token files, and recorded sampling parameters.

## Stage 4 token repair

The default token-repair profile is now:

```bash
PROFILE=h200_1024
```

It selects:

- `configs/stage4/self_corrupt/mmu_lora_train_token_repair_8x8_h200_1024_adamw.yaml`
- `configs/stage4/self_corrupt/mmu_probe_lora_token_repair_8x8_h200_1024.yaml`

The H200 training config disables gradient checkpointing and uses standard
AdamW. This removes the memory-saving overhead that was necessary for L40S.
It also validates every 3 epochs, plus the final epoch, to avoid spending H200
time on repeated full validation passes that do not update the adapter.

Stage-4 DDP submissions now default to a 24-hour Slurm walltime. Override it at
submit time when a shorter backfill slot is more useful:

```bash
TRAIN_TIME=12:00:00 MODE=submit_train bash scripts/training/run_stage4_token_repair_lora.sh
```

The training loop caches prepared image-token payloads and tokenized text within
each process so repeated epochs do not re-read pickle files or re-tokenize the
same prompts. Defaults are `ASCR_LORA_IMAGE_CACHE_SIZE=4096` and
`ASCR_LORA_TEXT_CACHE_SIZE=8192`; set either to `0` to disable that cache while
debugging memory behavior.

To force the old fallback:

```bash
PROFILE=l40s_1024_gc MODE=submit_train bash scripts/training/run_stage4_token_repair_lora.sh
```

After `training_manifest.json` exists, summarize measured training speed:

```bash
MODE=speed_report bash scripts/training/run_stage4_token_repair_lora.sh
```

If an older L40S manifest is available, include it as the baseline:

```bash
SPEED_BASELINE_MANIFESTS=outputs/path/to/l40s/training_manifest.json \
SPEED_BASELINE_LABELS=l40s_1024_gc \
SPEED_BASELINE_LABEL=l40s_1024_gc \
MODE=speed_report bash scripts/training/run_stage4_token_repair_lora.sh
```

The report is written under
`outputs/stage4_token_repair/repair_cells_8x8/speed_report/` and includes epoch
time, global samples/s, optimizer, gradient-checkpointing state, and speedup
relative to any baseline manifest included in the report.

## Stage 4 curriculum

The curriculum default is also `PROFILE=h200_1024`:

```bash
sbatch jobs/stage4/train_mmu_lora_curriculum.sbatch
```

Generated H200 grid configs live under `configs/stage4/self_corrupt/` and write
adapters to `lora_h200_1024px_adamw` directories. L40S configs are still present
for reproducibility and memory fallback.

## Runtime defaults

`jobs/stage4/train_mmu_lora_ddp.sbatch` now defaults to production-speed DDP
settings: reduced NCCL/Torch debug logging, no pre-collective debug barrier, and
static-graph DDP. Set the corresponding environment variables explicitly when
debugging distributed failures.

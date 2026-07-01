# Server AI Task: Stage 5/6 + Infra Script Pack

This document tracks the Windows Codex implementation of the 2026-06-28
remaining-scripts request.

## Coverage

### Phase 5 self-corruption loop

- `ascr/selectors/mmu_localizer_selector.py`
- `ascr/cli/stage5_self_corrupt_loop.py`
- `ascr/cli/stage5_self_corrupt_benchmark.py`
- `ascr/cli/stage5_compare_loop_results.py`
- `scripts/training/run_stage5_loop.sh`
- `jobs/stage5/self_corrupt_loop.sbatch`
- `configs/stage5/self_corrupt/ascr_loop_smoke.yaml`
- `configs/stage5/self_corrupt/benchmark_hard64.yaml`

### Phase 6 synthetic-to-real transfer

- `ascr/cli/stage6_transfer_probe.py`
- `ascr/cli/stage6_multi_arm_benchmark.py`
- `ascr/analysis/stage6_transfer_metrics.py`
- `configs/stage6/transfer_probe.yaml`

### Dataset scaling

- `ascr/cli/stage3_sample_prompts.py`
- `configs/stage3/self_corrupt/locality_probe_hard64.yaml`
- `configs/stage3/self_corrupt/locality_probe_hard256.yaml`
- `configs/stage3/self_corrupt/locality_probe_bench512.yaml`
- `configs/stage3/self_corrupt/locality_probe_bench1k.yaml`
- `scripts/training/run_stage3_scale_dataset.sh`

### Training infrastructure

- `ascr/training/stage4_mmu_lora_ddp.py`
- `ascr/cli/stage4_train_mmu_lora_ddp.py`
- `ascr/cli/stage4_batch_train.py`
- `ascr/cli/stage4_resume_training.py`
- `jobs/stage4/train_mmu_lora_ddp.sbatch`
- `jobs/stage4/stage4_multi_gpu_eval.sbatch`
- `ascr/cli/stage4_hyperparameter_search.py`
- `ascr/cli/stage4_adapter_registry.py`
- `ascr/cli/stage4_generate_config.py`
- `ascr/cli/stage4_merge_probe_shards.py`
- `scripts/training/run_stage4_recovery_submit.sh`
- `scripts/training/run_hard256_full_pipeline.sh`

### Analysis and routing

- `ascr/cli/stage4_cross_grid_compare.py`
- `ascr/analysis/stage4_failure_router.py`
- `ascr/cli/stage4_per_prompt_breakdown.py`

### Operations

- `scripts/slurm/dynamic_gpu_detect.sh`
- `scripts/slurm/qos_batch_submit.sh`
- `ascr/cli/server_dashboard.py`
- `ascr/cli/server_health_check.py`

### Integration

- `tests/test_stage3_4_5_integration.py`
- `scripts/training/run_stage3_to_stage5_e2e.sh`

## Smoke Commands

```bash
python -m unittest tests.test_stage3_4_5_integration
bash scripts/training/run_stage3_to_stage5_e2e.sh
```

## Server Commands

Stage-5 single prompt:

```bash
bash scripts/training/run_stage5_loop.sh
```

Stage-5 benchmark:

```bash
MODE=benchmark OUTPUT_DIR=outputs/stage5_self_corrupt/benchmark/hard64 LIMIT=16 \
  bash scripts/training/run_stage5_loop.sh
MODE=compare OUTPUT_DIR=outputs/stage5_self_corrupt/benchmark/hard64 \
  bash scripts/training/run_stage5_loop.sh
```

Stage-6 transfer smoke:

```bash
python -m ascr.cli.stage6_transfer_probe \
  --prompts configs/benchmarks/prompts/geneval_553.txt \
  --limit 32 \
  --config configs/stage6/transfer_probe.yaml \
  --output-dir outputs/stage6_transfer/geneval_smoke32
```

Stage-3 dataset expansion:

```bash
MODE=plan bash scripts/training/run_stage3_scale_dataset.sh
MODE=sample PROMPT_COUNT=256 bash scripts/training/run_stage3_scale_dataset.sh
MODE=submit PROMPT_COUNT=256 PROMPTS_PER_TASK=4 bash scripts/training/run_stage3_scale_dataset.sh
```

Stage-4 8-GPU launch entry:

```bash
sbatch jobs/stage4/train_mmu_lora_ddp.sbatch
```

The DDP sbatch now exports `PYTHONUNBUFFERED=1`, `NCCL_DEBUG=INFO`,
`NCCL_TIMEOUT=1800`, `NCCL_ASYNC_ERROR_HANDLING=1`, and
`TORCH_DISTRIBUTED_DEBUG=DETAIL` by default. Override them only when debugging
shows a better cluster-specific setting.

It also exports DDP options for large frozen-base LoRA training:
`ASCR_DDP_IGNORE_FROZEN=1`, `ASCR_DDP_INIT_SYNC=0`,
`ASCR_DDP_BROADCAST_BUFFERS=0`, `ASCR_DDP_FIND_UNUSED_PARAMETERS=0`, and
`ASCR_DDP_GRADIENT_AS_BUCKET_VIEW=1`. It also exports `ASCR_DDP_DEBUG=1` and
`ASCR_DDP_IGNORE_FROZEN_METHOD=attribute`. The sbatch now also defaults
`ASCR_DDP_PRE_COLLECTIVE_BARRIER=1`, inserting a debugged barrier before the
rank-consistency tensor gather. Every rank prints `ASCR_DDP_DEBUG` JSON markers
before and after the pre-collective barrier, rank-consistency gather,
frozen-parameter ignore pass, DDP option generation, and DDP constructor. If a
run hangs, preserve all of those lines in `docs/AI_COLLAB_LOG.md`.

Minimal DDP constructor smoke:

```bash
CONFIG=configs/stage4/self_corrupt/mmu_lora_train_hard256_grid4_vq_tokens_l40s_1024_gc_adam8bit.yaml \
DATA_JSONL=outputs/stage4_self_corrupt/mmu_lora_hard256_curriculum/grid4/vq_tokens/lumina_sft/train.jsonl \
VAL_JSONL=outputs/stage4_self_corrupt/mmu_lora_hard256_curriculum/grid4/vq_tokens/lumina_sft/val.jsonl \
NPROC=2 LIMIT=8 EPOCHS=1 ASCR_DDP_DEBUG=1 ASCR_DDP_PRE_COLLECTIVE_BARRIER=1 \
sbatch --partition=gpu_shared --gres=gpu:2 --cpus-per-task=16 --mem=180G \
  --time=01:00:00 --export=ALL,CONFIG,DATA_JSONL,VAL_JSONL,NPROC,LIMIT,EPOCHS,ASCR_DDP_DEBUG,ASCR_DDP_PRE_COLLECTIVE_BARRIER \
  jobs/stage4/train_mmu_lora_ddp.sbatch
```

Retest order after the 2026-06-30 local fix:

1. Run the 2-GPU NCCL smoke above. Success means logs show
   `rank_consistency_pre_collective_barrier_start`,
   `rank_consistency_pre_collective_barrier_done`,
   `rank_consistency_tensor_gather_done`, and then at least the first training
   step or epoch summary.
2. If NCCL still hangs at the pre-collective barrier or the following gather,
   rerun the same command with `ASCR_DDP_BACKEND=gloo` added to the environment.
   The local fix makes `_call_model_loss` try Python token rows before
   tensor-backed rows, which targets the previous GLOO crash:
   `TypeError: unsupported operand type(s) for +: 'Tensor' and 'list'`.
3. If 2-GPU GLOO reaches training, scale to `NPROC=8 --gres=gpu:8` with GLOO.
   This is expected to be slower than NCCL but should be acceptable for the
   LoRA parameter count. If both NCCL and GLOO fail, use the existing
   single-GPU/resume path while logging the exact final failure line.

Stage-4 multi-GPU eval shards:

```bash
sbatch jobs/stage4/stage4_multi_gpu_eval.sbatch
MODE=summarize OUTPUT_ROOT=outputs/stage4_self_corrupt/multi_gpu_eval/grid4_1024gc \
  bash scripts/training/run_stage4_multi_gpu_eval.sh
```

`GPU_IDS` can override auto-detection, for example `GPU_IDS=2,4,6,7`, when
Slurm exposes a non-contiguous or UUID-based allocation. `CHUNKS_PER_GPU=2`
splits each GPU's eval slice into smaller subprocesses, which is the preferred
retry path when a larger eval shard OOMs or times out.

Stage-4 grid batch train/probe in one allocation:

```bash
python -m ascr.cli.stage4_batch_train --grids 4,8,16
```

Hard256 full pipeline:

```bash
MODE=plan bash scripts/training/run_hard256_full_pipeline.sh
MODE=generate_configs bash scripts/training/run_hard256_full_pipeline.sh
MODE=prepare_sft bash scripts/training/run_hard256_full_pipeline.sh
MODE=check_inputs bash scripts/training/run_hard256_full_pipeline.sh
MODE=submit_train bash scripts/training/run_hard256_full_pipeline.sh
MODE=submit_eval bash scripts/training/run_hard256_full_pipeline.sh
MODE=submit_eval_recovery CHUNKS_PER_GPU=2 bash scripts/training/run_hard256_full_pipeline.sh
MODE=summarize bash scripts/training/run_hard256_full_pipeline.sh
MODE=registry bash scripts/training/run_hard256_full_pipeline.sh
```

Hard256 generated configs now use a 60/20/20 train/val/test split. The
pipeline's `prepare_sft` mode creates `lumina_sft/train.jsonl`,
`lumina_sft/val.jsonl`, and `lumina_sft/test.jsonl` per grid. Training uses
train plus val only: `val.jsonl` drives best-checkpoint selection and early
stopping, while test remains held out for final probe/reporting.

Stage-4 training fallback submit:

```bash
MODE=submit CONFIG=configs/stage4/self_corrupt/mmu_lora_train_hard256_grid4_vq_tokens_l40s_1024_gc_adam8bit.yaml \
  GPU_FALLBACKS="8 4 1" bash scripts/training/run_stage4_recovery_submit.sh
MODE=recover JOB_ID=<failed_job_id> \
  GPU_FALLBACKS="8 4 1" bash scripts/training/run_stage4_recovery_submit.sh
```

`run_stage4_recovery_submit.sh` writes `outputs/stage4_self_corrupt/recovery_submit_attempts.tsv`
so a later `MODE=recover JOB_ID=...` can infer whether the failed job was the
8-GPU, 4-GPU, or 1-GPU attempt. Use `DRY_RUN=1` to print the exact `sbatch`
command without submitting.

The recovery submit wrapper also passes `VAL_JSONL`,
`EARLY_STOPPING_PATIENCE`, and `EARLY_STOPPING_MIN_DELTA` through to the DDP
training job. Default patience is 3.

Stage-5 multi-prompt single-node run:

```bash
sbatch jobs/stage5/multi_prompt_loop.sbatch
MODE=summarize OUTPUT_ROOT=outputs/stage5_self_corrupt/multi_prompt \
  bash scripts/training/run_stage5_multi_prompt.sh
```

The Stage-5 multi-prompt wrapper also honors `GPU_IDS`, `GPU_COUNT`,
`PROMPT_OFFSET`, and `PROMPTS_PER_GPU`.

## Stage-5 Evaluation Design

Do not judge Stage-5 repaired images in isolation. Use paired A/B evaluation
with the same prompt, seed, and baseline VQ tokens whenever possible:

- `baseline`: original generation without Stage-5 repair.
- `ours`: Stage-5 predicted cells reopened and decoded.
- `random_repair`: reopen the same number of random 8x8 cells.
- `oracle_repair`: synthetic corruption only; reopen the GT 8x8 cells projected
  from the 64x64 token corruption mask.
- Optional `full_regeneration`: regenerate the whole image as a high-variance
  reference.

Report two separate evidence layers:

1. Synthetic localization metrics, using token-mask GT projected to 8x8 cells:
   hit-any, precision, recall, F1, IoU, over-reopen rate, false-empty rate, and
   clean false-positive rate. This layer does not require an external API.
2. Final image-quality metrics, using blinded paired VLM/API or human judges:
   `ours_win_rate`, `tie_rate`, `baseline_win_rate`, and
   `net_win_rate = ours_win_rate - baseline_win_rate`. Bucket results by
   corruption size, corruption operator, predicted cell count, and prompt
   source.

For API judges, randomize which image is A/B and require compact JSON only:

```json
{
  "winner": "A|B|tie",
  "prompt_alignment": "A|B|tie",
  "visual_quality": "A|B|tie",
  "artifact_reduction": "A|B|tie",
  "local_preservation": "A|B|tie",
  "reason": "short text"
}
```

External APIs are appropriate for the image-quality layer, but they should
answer a paired question: which image better satisfies the prompt while reducing
artifacts and preserving unchanged regions? Do not ask whether one image is
good in isolation. Keep a small human-review slice for close calls and API
judge disagreements.

## Current Boundaries

- `stage4_train_mmu_lora_ddp` now builds one Lumina+LoRA replica per rank,
  wraps it with `DistributedDataParallel`, uses `DistributedSampler`, and only
  saves from rank 0. It now also validates LoRA trainable parameter signatures
  across ranks before DDP wrapping and writes `ddp_rank_consistency_error.json`
  if PEFT injection differs by rank. It still needs server validation on a real
  8-GPU node.
- `stage4_resume_training` skips complete adapters and relaunches interrupted
  configs from the latest epoch-level adapter checkpoint when available.
- The current cluster has 100+ GPUs across many 8-GPU nodes; ASCR's operating
  philosophy is to request more resources for each suitable task and trade
  parallel GPU allocation for shorter wall-clock iteration time.
- Stage-6 `stage1_qwen` and `stage3_selector` arms are placeholder manifests in
  `stage6_multi_arm_benchmark`; plug mature runners into those arms when the
  server needs full comparisons.
- Stage-5 defaults to `share_engine: true` to avoid loading generator, MMU, and
  LoRA copies at the same time. The loop now reuses one
  `LuminaNativeEngine` instance and lazily attaches the LoRA adapter before
  the MMU answer call; it also defaults to
  `offload_generator_before_mmu: true`, which now calls
  `release_generation_cache()` to clear transient Python/CUDA cache while
  keeping the base model/tokenizer/VQ-VAE resident. Full unload/reload is no
  longer the default and only happens if `allow_full_unload_before_mmu: true`
  is explicitly set. This behavior is covered by
  `test_stage5_share_engine_reuses_one_lumina_instance_and_attaches_lora_lazily`.
  Set `share_engine: false` only when GPU memory is sufficient and separate
  generation/MMU behavior must be isolated.
- `run_hard256_full_pipeline.sh` now includes explicit `prepare_sft` and
  `check_inputs` modes. `submit_train` refuses to submit if the Hard256 dataset
  or per-grid Lumina SFT `train.jsonl`/`val.jsonl` files are missing, unless
  `SKIP_INPUT_CHECK=1` is set intentionally.
- Stage-4 multi-GPU eval supports per-GPU chunking through `CHUNKS_PER_GPU`;
  Hard256 `submit_eval_recovery` defaults to two chunks per GPU to retry failed
  evals with smaller per-process sample counts.
- Stage-4 train recovery records submission attempts and can infer the current
  fallback position from the failed Slurm job id before resubmitting with the
  next smaller GPU count.
- All Stage-5/6 CLIs support `--mock` for local wiring tests without loading
  Lumina.

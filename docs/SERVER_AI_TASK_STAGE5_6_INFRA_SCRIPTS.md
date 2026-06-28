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

Stage-4 multi-GPU eval shards:

```bash
sbatch jobs/stage4/stage4_multi_gpu_eval.sbatch
MODE=summarize OUTPUT_ROOT=outputs/stage4_self_corrupt/multi_gpu_eval/grid4_1024gc \
  bash scripts/training/run_stage4_multi_gpu_eval.sh
```

`GPU_IDS` can override auto-detection, for example `GPU_IDS=2,4,6,7`, when
Slurm exposes a non-contiguous or UUID-based allocation.

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
MODE=summarize bash scripts/training/run_hard256_full_pipeline.sh
MODE=registry bash scripts/training/run_hard256_full_pipeline.sh
```

Stage-4 training fallback submit:

```bash
MODE=submit CONFIG=configs/stage4/self_corrupt/mmu_lora_train_hard256_grid4_vq_tokens_l40s_1024_gc_adam8bit.yaml \
  GPU_FALLBACKS="8 4 1" bash scripts/training/run_stage4_recovery_submit.sh
```

Stage-5 multi-prompt single-node run:

```bash
sbatch jobs/stage5/multi_prompt_loop.sbatch
MODE=summarize OUTPUT_ROOT=outputs/stage5_self_corrupt/multi_prompt \
  bash scripts/training/run_stage5_multi_prompt.sh
```

The Stage-5 multi-prompt wrapper also honors `GPU_IDS`, `GPU_COUNT`,
`PROMPT_OFFSET`, and `PROMPTS_PER_GPU`.

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
  the MMU answer call; this behavior is covered by
  `test_stage5_share_engine_reuses_one_lumina_instance_and_attaches_lora_lazily`.
  Set `share_engine: false` only when GPU memory is sufficient and separate
  generation/MMU behavior must be isolated.
- `run_hard256_full_pipeline.sh` now includes explicit `prepare_sft` and
  `check_inputs` modes. `submit_train` refuses to submit if the Hard256 dataset
  or per-grid Lumina SFT `train.jsonl` files are missing, unless
  `SKIP_INPUT_CHECK=1` is set intentionally.
- All Stage-5/6 CLIs support `--mock` for local wiring tests without loading
  Lumina.

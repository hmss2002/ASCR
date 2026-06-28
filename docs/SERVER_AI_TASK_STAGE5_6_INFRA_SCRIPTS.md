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
- `jobs/stage4/train_mmu_lora_ddp.sbatch`
- `ascr/cli/stage4_hyperparameter_search.py`
- `ascr/cli/stage4_adapter_registry.py`
- `ascr/cli/stage4_generate_config.py`

### Analysis and routing

- `ascr/cli/stage4_cross_grid_compare.py`
- `ascr/analysis/stage4_failure_router.py`
- `ascr/cli/stage4_per_prompt_breakdown.py`

### Operations

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

## Current Boundaries

- `stage4_train_mmu_lora_ddp` is a torchrun-safe coordinator around the current
  single-process trainer. True DistributedSampler/gradient sync remains a
  follow-up implementation task.
- Stage-6 `stage1_qwen` and `stage3_selector` arms are placeholder manifests in
  `stage6_multi_arm_benchmark`; plug mature runners into those arms when the
  server needs full comparisons.
- All Stage-5/6 CLIs support `--mock` for local wiring tests without loading
  Lumina.


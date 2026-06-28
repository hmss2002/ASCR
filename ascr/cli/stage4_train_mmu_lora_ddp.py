"""Torchrun entry point for Stage-4 MMU LoRA training."""

from __future__ import annotations

from ascr.training.stage4_mmu_lora_ddp import run_rank0_coordinated


def main(argv=None):
    return run_rank0_coordinated(argv)


if __name__ == "__main__":
    raise SystemExit(main())


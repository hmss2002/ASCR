"""Torchrun entry point for real Stage-4 MMU LoRA DDP training."""

from __future__ import annotations

from ascr.training.stage4_mmu_lora_ddp import train_lumina_lora_ddp


def main(argv=None):
    train_lumina_lora_ddp(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

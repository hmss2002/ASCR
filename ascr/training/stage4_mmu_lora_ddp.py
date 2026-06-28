"""Torchrun-compatible Stage-4 MMU LoRA training entry helpers.

The current Lumina LoRA trainer is single-process. This module gives the server
an explicit torchrun entry point and prevents multiple ranks from racing on the
same adapter directory. Rank 0 runs the existing trainer; other ranks wait at a
barrier when torch.distributed is available.
"""

from __future__ import annotations

import os

from ascr.cli.stage4_train_mmu_lora import main as train_main


def distributed_env():
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def run_rank0_coordinated(argv=None):
    env = distributed_env()
    if env["world_size"] <= 1:
        return train_main(argv)
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend=os.environ.get("ASCR_DDP_BACKEND", "nccl"))
        if env["rank"] == 0:
            code = train_main(argv)
        else:
            code = 0
        dist.barrier()
        return code
    except Exception:
        if env["rank"] == 0:
            return train_main(argv)
        return 0


"""Real torchrun/DDP training for Stage-4 Lumina MMU LoRA."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

from ascr.cli.stage4_train_mmu_lora import build_parser as build_stage4_parser
from ascr.core.config import load_config
from ascr.training.train_lumina_lora_smoke import (
    _build_optimizer,
    _call_model_loss,
    _configure_gradient_checkpointing,
    _dtype_value,
    _jsonl_rows,
    _load_training_stack,
    _prepare_lumina_lora_batch,
    build_parser as build_lora_parser,
)


def distributed_env():
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def _resolve_lora_args(argv=None):
    args = build_stage4_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    lora_args = build_lora_parser().parse_args([])
    for key, value in config.items():
        setattr(lora_args, key.replace("-", "_"), value)
    if args.data_jsonl:
        lora_args.data_jsonl = args.data_jsonl
    if args.output_dir:
        lora_args.output_dir = args.output_dir
    for key in (
        "epochs",
        "limit",
        "optimizer",
        "image_size",
        "max_seq_len",
        "target_modules",
        "torch_dtype",
        "device_map",
        "gradient_checkpointing",
        "gradient_checkpointing_fallback",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(lora_args, key, value)
    lora_args.device_map = "none"
    return lora_args


def _ddp_loss(torch, model, input_ids, labels, device):
    return _call_model_loss(torch, model, input_ids, labels, device=device)


def train_lumina_lora_ddp(argv=None):
    args = _resolve_lora_args(argv)
    torch, LoraConfig, TaskType, get_peft_model, AutoTokenizer, model_cls, add_break_line = _load_training_stack(args.repo_path)
    env = distributed_env()
    if env["world_size"] <= 1:
        from ascr.training.train_lumina_lora_smoke import train_lumina_lora_smoke

        return train_lumina_lora_smoke(args)

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler

    backend = os.environ.get("ASCR_DDP_BACKEND") or ("nccl" if torch.cuda.is_available() else "gloo")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(env["local_rank"])
        device = torch.device("cuda", env["local_rank"])
    else:
        device = torch.device("cpu")
    random.seed(int(args.seed) + env["rank"])

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    model = model_cls.from_pretrained(args.checkpoint_path, torch_dtype=_dtype_value(torch, args.torch_dtype))
    model = model.to(device)
    gradient_checkpointing_report = {"requested": bool(args.gradient_checkpointing), "backend": "disabled", "wrapped_module_count": 0}
    if args.gradient_checkpointing:
        gradient_checkpointing_report = _configure_gradient_checkpointing(
            torch,
            model,
            mode=args.gradient_checkpointing_fallback,
        )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=[part.strip() for part in str(args.target_modules).split(",") if part.strip()],
    )
    model = get_peft_model(model, lora_config).to(device)
    model.train()
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[env["local_rank"]] if device.type == "cuda" else None,
        output_device=env["local_rank"] if device.type == "cuda" else None,
        find_unused_parameters=True,
    )
    rows = list(_jsonl_rows(args.data_jsonl))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError(f"No training rows found in {args.data_jsonl}")
    sampler = DistributedSampler(rows, num_replicas=env["world_size"], rank=env["rank"], shuffle=True, seed=int(args.seed))
    optimizer = _build_optimizer(
        torch,
        args.optimizer,
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    local_losses = []
    for epoch in range(int(args.epochs)):
        sampler.set_epoch(epoch)
        total = 0.0
        count = 0
        for step, row_index in enumerate(sampler):
            input_ids, labels = _prepare_lumina_lora_batch(
                torch,
                tokenizer,
                add_break_line,
                rows[int(row_index)],
                args,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _ddp_loss(torch, ddp_model, input_ids, labels, device=device)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().item())
            total += value
            count += 1
            local_losses.append({"rank": env["rank"], "epoch": epoch, "step": step, "row_index": int(row_index), "loss": value})
        local_avg = total / max(1, count)
        avg_tensor = torch.tensor([local_avg], dtype=torch.float32, device=device)
        dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
        avg_tensor /= env["world_size"]
        if env["rank"] == 0:
            print(f"epoch {epoch}: ddp_avg_loss={float(avg_tensor.item()):.6f}")
    gathered_losses = [None for _ in range(env["world_size"])]
    dist.all_gather_object(gathered_losses, local_losses)
    manifest = None
    if env["rank"] == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ddp_model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        losses = [item for rank_losses in gathered_losses for item in (rank_losses or [])]
        losses.sort(key=lambda item: (item["epoch"], item["step"], item["rank"]))
        manifest = {
            "schema_version": "ascr.lumina_lora_ddp.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "data_jsonl": str(args.data_jsonl),
            "checkpoint_path": str(args.checkpoint_path),
            "repo_path": str(args.repo_path),
            "output_dir": str(output_dir),
            "row_count": len(rows),
            "epochs": int(args.epochs),
            "world_size": env["world_size"],
            "backend": backend,
            "optimizer": str(args.optimizer),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "image_size": int(args.image_size),
            "max_seq_len": int(args.max_seq_len),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "torch_dtype": str(args.torch_dtype),
            "device_map": "none",
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "gradient_checkpointing_report": gradient_checkpointing_report,
            "losses": losses,
            "final_loss": losses[-1]["loss"] if losses else None,
        }
        (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return manifest

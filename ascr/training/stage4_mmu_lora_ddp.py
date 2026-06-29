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
    _apply_lora_adapter,
    _build_optimizer,
    _call_model_loss,
    _configure_gradient_checkpointing,
    _dtype_value,
    _lora_parameter_report,
    _jsonl_rows,
    _load_training_stack,
    _prepare_lumina_lora_batch,
    _save_epoch_checkpoint,
    _trainable_parameters,
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
        "resume_from_adapter",
        "checkpoint_every_epochs",
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


def _assert_rank_consistent_lora(dist, env, report, output_dir):
    reports = [None for _ in range(env["world_size"])]
    dist.all_gather_object(reports, report)
    keys = ("trainable_tensor_count", "trainable_parameter_count", "lora_tensor_count", "lora_trainable_tensor_count")
    expected = {key: reports[0].get(key) for key in keys}
    mismatches = [
        {"rank": index, "report": item}
        for index, item in enumerate(reports)
        if any(item.get(key) != expected[key] for key in keys)
    ]
    if mismatches:
        payload = {
            "schema_version": "ascr.lumina_lora_ddp.rank_consistency_error.v1",
            "expected": expected,
            "reports": reports,
            "mismatches": mismatches,
        }
        if env["rank"] == 0:
            path = Path(output_dir) / "ddp_rank_consistency_error.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"PEFT/DDP rank parameter mismatch before DDP wrapping: {json.dumps(payload, sort_keys=True)}")
    if report.get("trainable_tensor_count", 0) <= 0 or report.get("lora_trainable_tensor_count", 0) <= 0:
        raise RuntimeError(f"Rank {env['rank']} has no trainable LoRA tensors before DDP wrapping: {json.dumps(report, sort_keys=True)}")
    return reports


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
    model, lora_parameter_report = _apply_lora_adapter(LoraConfig, TaskType, get_peft_model, model, args)
    model = model.to(device)
    gradient_checkpointing_report = {"requested": bool(args.gradient_checkpointing), "backend": "disabled", "wrapped_module_count": 0}
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        gradient_checkpointing_report = _configure_gradient_checkpointing(
            torch,
            model,
            mode=args.gradient_checkpointing_fallback,
        )
    lora_parameter_report = _lora_parameter_report(model)
    lora_parameter_report["resume_from_adapter"] = str(args.resume_from_adapter) if args.resume_from_adapter else None
    rank_lora_reports = _assert_rank_consistent_lora(dist, env, lora_parameter_report, args.output_dir)
    model.train()
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[env["local_rank"]] if device.type == "cuda" else None,
        output_device=env["local_rank"] if device.type == "cuda" else None,
        find_unused_parameters=False,
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
        _trainable_parameters(ddp_model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    local_losses = []
    checkpoints = []
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
            if int(args.checkpoint_every_epochs or 0) > 0 and (epoch + 1) % int(args.checkpoint_every_epochs) == 0:
                checkpoints.append(_save_epoch_checkpoint(ddp_model.module, tokenizer, args.output_dir, epoch + 1))
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
            "lora_parameter_report": lora_parameter_report,
            "rank_lora_reports": rank_lora_reports,
            "resume_from_adapter": str(args.resume_from_adapter) if args.resume_from_adapter else None,
            "checkpoint_every_epochs": int(args.checkpoint_every_epochs or 0),
            "checkpoints": checkpoints,
            "losses": losses,
            "final_loss": losses[-1]["loss"] if losses else None,
        }
        (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return manifest

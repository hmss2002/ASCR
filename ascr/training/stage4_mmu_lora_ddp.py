"""Real torchrun/DDP training for Stage-4 Lumina MMU LoRA."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
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
    _copy_checkpoint_to_output,
    _dtype_value,
    _lora_parameter_report,
    _jsonl_rows,
    _load_training_stack,
    _mean_lora_loss,
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
    if args.val_jsonl:
        lora_args.val_jsonl = args.val_jsonl
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
        "early_stopping_patience",
        "early_stopping_min_delta",
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


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _mark_ddp_ignored_frozen_parameters(DistributedDataParallel, model):
    if not _env_bool("ASCR_DDP_IGNORE_FROZEN", default=True):
        return {"enabled": False, "ignored_parameter_count": 0, "ignored_parameter_names_sample": []}
    ignored = [
        name
        for name, parameter in model.named_parameters()
        if not getattr(parameter, "requires_grad", False)
    ]
    setter = getattr(DistributedDataParallel, "_set_params_and_buffers_to_ignore_for_model", None)
    if callable(setter) and ignored:
        setter(model, ignored)
    elif ignored:
        model._ddp_params_and_buffers_to_ignore = set(ignored)
        for name, parameter in model.named_parameters():
            if name in model._ddp_params_and_buffers_to_ignore:
                parameter._ddp_ignored = True
    return {
        "enabled": True,
        "ignored_parameter_count": len(ignored),
        "ignored_parameter_names_sample": ignored[:32],
    }


def _ddp_constructor_options(DistributedDataParallel, env, device):
    options = {
        "device_ids": [env["local_rank"]] if device.type == "cuda" else None,
        "output_device": env["local_rank"] if device.type == "cuda" else None,
        "find_unused_parameters": _env_bool("ASCR_DDP_FIND_UNUSED_PARAMETERS", default=False),
        "broadcast_buffers": _env_bool("ASCR_DDP_BROADCAST_BUFFERS", default=False),
        "gradient_as_bucket_view": _env_bool("ASCR_DDP_GRADIENT_AS_BUCKET_VIEW", default=True),
    }
    bucket_cap_mb = os.environ.get("ASCR_DDP_BUCKET_CAP_MB")
    if bucket_cap_mb:
        options["bucket_cap_mb"] = int(bucket_cap_mb)
    if "ASCR_DDP_STATIC_GRAPH" in os.environ:
        options["static_graph"] = _env_bool("ASCR_DDP_STATIC_GRAPH", default=False)
    if "ASCR_DDP_INIT_SYNC" in os.environ:
        options["init_sync"] = _env_bool("ASCR_DDP_INIT_SYNC", default=False)
    else:
        options["init_sync"] = False

    supported = set(inspect.signature(DistributedDataParallel).parameters)
    filtered = {key: value for key, value in options.items() if key in supported}
    ignored = sorted(set(options) - set(filtered))
    return filtered, ignored


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
    ddp_ignore_report = _mark_ddp_ignored_frozen_parameters(DistributedDataParallel, model)
    ddp_options, ddp_ignored_options = _ddp_constructor_options(DistributedDataParallel, env, device)
    if env["rank"] == 0:
        print(
            "constructing DDP with "
            f"options={json.dumps(ddp_options, sort_keys=True, default=str)}, "
            f"ignored_options={ddp_ignored_options}, "
            f"ignored_frozen_parameters={ddp_ignore_report['ignored_parameter_count']}",
            flush=True,
        )
    ddp_model = DistributedDataParallel(model, **ddp_options)
    rows = list(_jsonl_rows(args.data_jsonl))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError(f"No training rows found in {args.data_jsonl}")
    val_rows = list(_jsonl_rows(args.val_jsonl)) if args.val_jsonl else []
    if args.limit is not None and val_rows:
        val_rows = val_rows[: int(args.limit)]
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
    best_val_loss = None
    best_checkpoint = None
    epochs_without_improvement = 0
    stopped_early = False
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
        val_loss = None
        if val_rows:
            local_val_rows = val_rows[env["rank"]::env["world_size"]]
            local_val_loss = _mean_lora_loss(
                torch,
                ddp_model,
                tokenizer,
                add_break_line,
                local_val_rows,
                args,
                device=device,
            )
            local_val_count = len(local_val_rows)
            local_val_total = float(local_val_loss or 0.0) * local_val_count
            val_tensor = torch.tensor([local_val_total, float(local_val_count)], dtype=torch.float32, device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            val_loss = float((val_tensor[0] / val_tensor[1].clamp_min(1.0)).item())
        stop_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        if env["rank"] == 0:
            suffix = f", val_loss={val_loss:.6f}" if val_loss is not None else ""
            print(f"epoch {epoch}: ddp_avg_loss={float(avg_tensor.item()):.6f}{suffix}", flush=True)
            should_save_epoch = int(args.checkpoint_every_epochs or 0) > 0 and (epoch + 1) % int(args.checkpoint_every_epochs) == 0
            if val_loss is not None:
                improved = best_val_loss is None or val_loss < best_val_loss - float(args.early_stopping_min_delta)
                if improved:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_checkpoint = _save_epoch_checkpoint(ddp_model.module, tokenizer, args.output_dir, epoch + 1)
                    checkpoints.append(best_checkpoint)
                else:
                    epochs_without_improvement += 1
                if int(args.early_stopping_patience or 0) > 0 and epochs_without_improvement >= int(args.early_stopping_patience):
                    stopped_early = True
                    stop_tensor[0] = 1
                    print(
                        f"early stopping at epoch {epoch}: best_val_loss={best_val_loss:.6f}, "
                        f"patience={int(args.early_stopping_patience)}",
                        flush=True,
                    )
            elif should_save_epoch:
                checkpoints.append(_save_epoch_checkpoint(ddp_model.module, tokenizer, args.output_dir, epoch + 1))
        dist.broadcast(stop_tensor, src=0)
        if int(stop_tensor.item()) == 1:
            break
    gathered_losses = [None for _ in range(env["world_size"])]
    dist.all_gather_object(gathered_losses, local_losses)
    manifest = None
    if env["rank"] == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if best_checkpoint:
            _copy_checkpoint_to_output(best_checkpoint, output_dir)
        else:
            ddp_model.module.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        losses = [item for rank_losses in gathered_losses for item in (rank_losses or [])]
        losses.sort(key=lambda item: (item["epoch"], item["step"], item["rank"]))
        manifest = {
            "schema_version": "ascr.lumina_lora_ddp.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "data_jsonl": str(args.data_jsonl),
            "val_jsonl": str(args.val_jsonl) if args.val_jsonl else None,
            "checkpoint_path": str(args.checkpoint_path),
            "repo_path": str(args.repo_path),
            "output_dir": str(output_dir),
            "row_count": len(rows),
            "epochs": int(args.epochs),
            "completed_epochs": max((item["epoch"] for item in losses), default=-1) + 1,
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
            "ddp_constructor_options": {key: str(value) for key, value in ddp_options.items()},
            "ddp_ignored_constructor_options": ddp_ignored_options,
            "ddp_ignore_report": ddp_ignore_report,
            "resume_from_adapter": str(args.resume_from_adapter) if args.resume_from_adapter else None,
            "checkpoint_every_epochs": int(args.checkpoint_every_epochs or 0),
            "early_stopping_patience": int(args.early_stopping_patience or 0),
            "early_stopping_min_delta": float(args.early_stopping_min_delta),
            "stopped_early": stopped_early,
            "best_val_loss": best_val_loss,
            "best_checkpoint": best_checkpoint,
            "checkpoints": checkpoints,
            "losses": losses,
            "final_loss": losses[-1]["loss"] if losses else None,
        }
        (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return manifest

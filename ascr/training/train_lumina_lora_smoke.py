import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import importlib
import json
import math
import os
import pickle
from pathlib import Path
import random
import shutil
import time

from ascr.core.peft_compat import ensure_transformers_tensor_parallel_compat


SP = {
    "mask": 126336,
    "newline": 126084,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,
    "eoi": 126350,
    "padding": 126339,
}

_IMAGE_TOKEN_CACHE = OrderedDict()
_TEXT_TOKEN_CACHE = OrderedDict()


def _cache_limit(name, default):
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _cache_get(cache, key):
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _cache_set(cache, key, value, limit):
    if int(limit) <= 0:
        return value
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > int(limit):
        cache.popitem(last=False)
    return value


def _clear_lumina_lora_caches():
    _IMAGE_TOKEN_CACHE.clear()
    _TEXT_TOKEN_CACHE.clear()


def _lumina_lora_cache_report():
    return {
        "image_token_cache_size": len(_IMAGE_TOKEN_CACHE),
        "image_token_cache_limit": _cache_limit("ASCR_LORA_IMAGE_CACHE_SIZE", 4096),
        "text_token_cache_size": len(_TEXT_TOKEN_CACHE),
        "text_token_cache_limit": _cache_limit("ASCR_LORA_TEXT_CACHE_SIZE", 8192),
    }


def _build_optimizer(torch, optimizer_name, parameters, lr, weight_decay):
    name = str(optimizer_name or "adamw").strip().lower().replace("-", "_")
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))
    if name in {"adamw8bit", "adamw_8bit", "8bit_adamw"}:
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError(
                "optimizer=adamw8bit requires bitsandbytes on the training server. "
                "Install it in .venv-lumina or use optimizer=adamw."
            ) from exc
        return bnb.optim.AdamW8bit(parameters, lr=float(lr), weight_decay=float(weight_decay))
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _mask_codes(codes, mode="random", force_mask=False):
    if not codes:
        return [], []
    if force_mask or mode == "all":
        indices = list(range(len(codes)))
    elif mode != "random":
        raise ValueError(f"Unknown answer mask mode: {mode}")
    elif len(codes) <= 5:
        mask_ratio = 1.0
        count = len(codes)
        indices = random.sample(range(len(codes)), count)
    else:
        ratio_seed = random.uniform(0, 1)
        mask_ratio = math.cos(ratio_seed * math.pi / 2)
        count = max(1, int(len(codes) * mask_ratio))
        indices = random.sample(range(len(codes)), count)
    masked = list(codes)
    labels = [-100] * len(codes)
    for index in indices:
        labels[index] = codes[index]
        masked[index] = SP["mask"]
    return masked, labels


def _jsonl_rows(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _center_crop_tokens(tokens, height, width, image_size):
    token_h = int(height) // 16
    token_w = int(width) // 16
    new_h = new_w = int(image_size) // 16
    if new_h >= token_h or new_w >= token_w:
        return list(tokens), token_h, token_w
    start_h = (token_h - new_h) // 2
    start_w = (token_w - new_w) // 2
    cropped = []
    for row in range(start_h, start_h + new_h):
        start = row * token_w + start_w
        cropped.extend(tokens[start:start + new_w])
    return cropped, new_h, new_w


def _load_training_stack(repo_path):
    import sys

    repo = str(Path(repo_path).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    ensure_transformers_tensor_parallel_compat()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer
    from model import LLaDAForMultiModalGeneration
    from utils.image_utils import add_break_line

    return torch, LoraConfig, TaskType, get_peft_model, AutoTokenizer, LLaDAForMultiModalGeneration, add_break_line


def _dtype_value(torch, dtype_name):
    dtype_name = str(dtype_name).lower()
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return dtype_map[dtype_name]


def _normalise_device(value):
    if value is None:
        return None
    text = str(value)
    if text in {"disk", "cpu"}:
        return text
    if text.isdigit():
        return f"cuda:{text}"
    return text


def _model_input_device(model, fallback="cpu"):
    device_map = getattr(model, "hf_device_map", None) or getattr(getattr(model, "base_model", None), "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            device = _normalise_device(value)
            if device and device not in {"cpu", "disk"}:
                return device
    try:
        return next(model.parameters()).device
    except Exception:
        return fallback


def _extract_loss(output):
    if hasattr(output, "loss"):
        return output.loss
    if isinstance(output, (list, tuple)) and output:
        return output[0]
    return output


def _as_tensor_batch(torch, values, device):
    if hasattr(values, "to"):
        return values.to(device=device, dtype=torch.long)
    return torch.tensor([values], dtype=torch.long, device=device)


def _as_token_row(values):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        values = values[0]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [int(value) for value in values]


def _call_model_loss(torch, model, input_ids, labels, device=None):
    device = device or _model_input_device(model)
    input_row = _as_token_row(input_ids)
    label_row = _as_token_row(labels)
    input_tensor = _as_tensor_batch(torch, input_ids, device)
    label_tensor = _as_tensor_batch(torch, labels, device)
    candidates = [
        ([input_row], [label_row]),
        ([input_tensor[0]], [label_tensor[0]]),
        (input_tensor, label_tensor),
    ]
    last_type_error = None
    for candidate_input_ids, candidate_labels in candidates:
        try:
            return _extract_loss(model(input_ids=candidate_input_ids, labels=candidate_labels))
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    return _extract_loss(model(input_ids=input_ids, labels=labels))


def _lora_parameter_report(model, sample_limit=32):
    trainable = []
    lora = []
    for name, parameter in model.named_parameters():
        if getattr(parameter, "requires_grad", False):
            trainable.append((name, int(parameter.numel())))
        if "lora_" in name.lower():
            lora.append((name, int(parameter.numel()), bool(getattr(parameter, "requires_grad", False))))
    return {
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(size for _name, size in trainable),
        "lora_tensor_count": len(lora),
        "lora_trainable_tensor_count": sum(1 for _name, _size, trainable_flag in lora if trainable_flag),
        "lora_parameter_count": sum(size for _name, size, _trainable_flag in lora),
        "trainable_names_sample": [name for name, _size in trainable[:sample_limit]],
        "lora_names_sample": [name for name, _size, _trainable_flag in lora[:sample_limit]],
    }


def _force_lora_trainable(model):
    changed = 0
    for name, parameter in model.named_parameters():
        if "lora_" in name.lower() and not getattr(parameter, "requires_grad", False):
            parameter.requires_grad_(True)
            changed += 1
    return changed


def _trainable_parameters(model):
    params = [parameter for parameter in model.parameters() if getattr(parameter, "requires_grad", False)]
    if not params:
        report = _lora_parameter_report(model)
        raise RuntimeError(f"No trainable LoRA parameters found after PEFT setup: {json.dumps(report, sort_keys=True)}")
    return params


def _apply_lora_adapter(LoraConfig, TaskType, get_peft_model, model, args):
    resume_from_adapter = getattr(args, "resume_from_adapter", None)
    if resume_from_adapter:
        ensure_transformers_tensor_parallel_compat()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resume_from_adapter), is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=[part.strip() for part in str(args.target_modules).split(",") if part.strip()],
        )
        model = get_peft_model(model, lora_config)
    forced_trainable_count = _force_lora_trainable(model)
    report = _lora_parameter_report(model)
    report["resume_from_adapter"] = str(resume_from_adapter) if resume_from_adapter else None
    report["forced_trainable_count"] = forced_trainable_count
    if report["trainable_tensor_count"] <= 0 or report["lora_trainable_tensor_count"] <= 0:
        raise RuntimeError(f"PEFT setup produced no trainable LoRA tensors: {json.dumps(report, sort_keys=True)}")
    return model, report


def _save_epoch_checkpoint(model, tokenizer, output_dir, epoch):
    checkpoint_dir = Path(output_dir) / "checkpoints" / f"epoch_{int(epoch):04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return str(checkpoint_dir)


def _copy_checkpoint_to_output(checkpoint_dir, output_dir):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in checkpoint_dir.iterdir():
        target = output_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _mean_lora_loss(torch, model, tokenizer, add_break_line, rows, args, device=None):
    if not rows:
        return None
    was_training = bool(getattr(model, "training", False))
    model.eval()
    total = 0.0
    with torch.no_grad():
        for row in rows:
            input_ids, labels = _prepare_lumina_lora_batch(
                torch,
                tokenizer,
                add_break_line,
                row,
                args,
                device=device,
            )
            loss = _call_model_loss(torch, model, input_ids, labels, device=device)
            if hasattr(loss, "detach"):
                value = loss.detach().item()
            elif hasattr(loss, "item"):
                value = loss.item()
            else:
                value = loss
            total += float(value)
    if was_training:
        model.train()
    return total / max(1, len(rows))


def _progress_bar_enabled(args):
    return bool(getattr(args, "progress_bar", True))


def _progress_every_steps(args):
    value = int(getattr(args, "progress_every_steps", 25) or 0)
    return max(0, value)


def _validation_every_epochs(args):
    value = int(getattr(args, "validation_every_epochs", 1) or 0)
    return max(0, value)


def _should_validate_epoch(args, epoch, total_epochs):
    if _validation_every_epochs(args) <= 0:
        return False
    if int(epoch) == int(total_epochs) - 1:
        return True
    return (int(epoch) + 1) % _validation_every_epochs(args) == 0


def _iter_with_progress(iterable, total, desc, args, enabled=True):
    if not enabled or not _progress_bar_enabled(args):
        return iterable, None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable, None
    bar = tqdm(
        iterable,
        total=int(total),
        desc=desc,
        unit="step",
        dynamic_ncols=True,
        mininterval=5.0,
    )
    return bar, bar


def _log_training_progress(epoch, step, total_steps, loss, total_loss, count, started_at, args, prefix="train"):
    every = _progress_every_steps(args)
    if every <= 0:
        return
    step_number = int(step) + 1
    total_steps = max(1, int(total_steps))
    if step_number != total_steps and step_number % every != 0:
        return
    elapsed_s = time.monotonic() - started_at
    samples_per_s = float(count) / max(elapsed_s, 1e-9)
    remaining_steps = max(0, total_steps - step_number)
    eta_s = remaining_steps / max(samples_per_s, 1e-9)
    avg_loss = float(total_loss) / max(1, int(count))
    print(
        f"{prefix} epoch={int(epoch)} step={step_number}/{total_steps} "
        f"loss={float(loss):.6f} avg_loss={avg_loss:.6f} "
        f"samples_per_s={samples_per_s:.4f} elapsed_s={elapsed_s:.1f} eta_s={eta_s:.1f}",
        flush=True,
    )


def _prepare_lumina_lora_batch(torch, tokenizer, add_break_line, row, args, device=None):
    image_tokens = _prepared_image_tokens(row["user_image"], args.image_size, add_break_line)
    instruction = "<system>" + row["system_prompt"] + "</system>" + "<user>" + row["user_prompt"] + "</user>"
    inst_ids = _cached_token_ids(
        tokenizer,
        instruction,
        int(args.prompt_max_length),
    )
    inst_ids = inst_ids[:-1] + [SP["boi"]] + image_tokens + [SP["eoi"]] + inst_ids[-1:]
    answer_ids = _cached_token_ids(
        tokenizer,
        row["answer_text"] + "</answer>",
        int(args.answer_max_length),
    )
    answer_ids, answer_labels = _mask_codes(answer_ids, mode=args.answer_mask_mode)
    pad_len = max(0, int(args.answer_max_length) - len(answer_ids))
    if args.ignore_pad_labels:
        pad_ids = [SP["padding"]] * pad_len
        pad_labels = [-100] * pad_len
    else:
        pad_ids, pad_labels = _mask_codes([SP["padding"]] * pad_len, mode=args.answer_mask_mode, force_mask=True)
    input_ids = inst_ids + [SP["answer_start"]] + answer_ids + pad_ids
    labels = [-100] * len(inst_ids) + [-100] + answer_labels + pad_labels
    if len(input_ids) > int(args.max_seq_len):
        input_ids = input_ids[: int(args.max_seq_len)]
        labels = labels[: int(args.max_seq_len)]
    else:
        padding = int(args.max_seq_len) - len(input_ids)
        input_ids += [SP["padding"]] * padding
        labels += [-100] * padding
    if device is None:
        return input_ids, labels
    return (
        torch.tensor([input_ids], dtype=torch.long, device=device),
        torch.tensor([labels], dtype=torch.long, device=device),
    )


def _prepared_image_tokens(user_image, image_size, add_break_line):
    key = (str(user_image), int(image_size), id(add_break_line))
    limit = _cache_limit("ASCR_LORA_IMAGE_CACHE_SIZE", 4096)
    cached = _cache_get(_IMAGE_TOKEN_CACHE, key) if limit > 0 else None
    if cached is not None:
        return cached
    with open(user_image, "rb") as handle:
        image_payload = pickle.load(handle)
    image_tokens, token_h, token_w = _center_crop_tokens(
        image_payload["input_ids"],
        image_payload["height"],
        image_payload["width"],
        image_size,
    )
    image_tokens = add_break_line(image_tokens, token_h, token_w, new_number=SP["newline"])
    return _cache_set(_IMAGE_TOKEN_CACHE, key, image_tokens, limit)


def _cached_token_ids(tokenizer, text, max_length):
    key = (id(tokenizer), str(text), int(max_length))
    limit = _cache_limit("ASCR_LORA_TEXT_CACHE_SIZE", 8192)
    cached = _cache_get(_TEXT_TOKEN_CACHE, key) if limit > 0 else None
    if cached is not None:
        return cached
    ids = tokenizer(
        text,
        truncation=True,
        max_length=int(max_length),
        padding=False,
        return_tensors="pt",
    ).input_ids[0].tolist()
    return _cache_set(_TEXT_TOKEN_CACHE, key, ids, limit)


def _is_tensor_like(value):
    return hasattr(value, "requires_grad") and hasattr(value, "detach")


def _is_checkpoint_candidate(name, module):
    if not name or not hasattr(module, "forward"):
        return False
    lowered_name = name.lower()
    class_name = module.__class__.__name__.lower()
    if "decoderlayer" in class_name or "transformerblock" in class_name:
        return True
    if "llada" in class_name and "block" in class_name:
        return True
    leaf = lowered_name.rsplit(".", 1)[-1]
    in_transformer_stack = any(marker in lowered_name for marker in (".layers.", ".blocks.", ".h."))
    return leaf.isdigit() and in_transformer_stack


def _checkpoint_function(torch):
    if not hasattr(torch, "utils") or not hasattr(torch.utils, "checkpoint"):
        importlib.import_module("torch.utils.checkpoint")
    return torch.utils.checkpoint.checkpoint


def _wrap_forward_with_checkpoint(torch, module):
    if getattr(module, "_ascr_gradient_checkpoint_wrapped", False):
        return False
    original_forward = module.forward
    checkpoint = _checkpoint_function(torch)

    def checkpointed_forward(*args, **kwargs):
        if not getattr(module, "training", False) or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        if not any(_is_tensor_like(arg) for arg in args):
            return original_forward(*args, **kwargs)

        def custom_forward(*inner_args):
            return original_forward(*inner_args, **kwargs)

        return checkpoint(custom_forward, *args, use_reentrant=False)

    module._ascr_original_forward = original_forward
    module._ascr_gradient_checkpoint_wrapped = True
    module.gradient_checkpointing = True
    module.forward = checkpointed_forward
    return True


def _enable_ascr_gradient_checkpointing(torch, model):
    wrapped = []
    for name, module in model.named_modules():
        if _is_checkpoint_candidate(name, module) and _wrap_forward_with_checkpoint(torch, module):
            wrapped.append(name)
    return {
        "backend": "ascr_module_wrapper",
        "wrapped_module_count": len(wrapped),
        "wrapped_modules": wrapped[:128],
        "wrapped_modules_truncated": len(wrapped) > 128,
    }


def _configure_gradient_checkpointing(torch, model, mode="auto"):
    mode = str(mode or "auto").strip().lower().replace("-", "_")
    if mode not in {"auto", "off", "force"}:
        raise ValueError(f"Unsupported gradient checkpointing fallback mode: {mode}")
    report = {
        "requested": True,
        "backend": "disabled",
        "wrapped_module_count": 0,
        "error": None,
    }
    if mode != "force":
        try:
            model.gradient_checkpointing_enable()
            report["backend"] = "huggingface"
            report["wrapped_module_count"] = None
            return report
        except (AttributeError, ValueError, NotImplementedError) as exc:
            report["error"] = str(exc)
            if mode == "off":
                print(f"warning: gradient_checkpointing not supported ({exc}); continuing without it")
                return report
    fallback = _enable_ascr_gradient_checkpointing(torch, model)
    report.update(fallback)
    if report["wrapped_module_count"] <= 0:
        raise RuntimeError(
            "gradient_checkpointing was requested but neither HuggingFace support nor "
            "the ASCR fallback wrapper found decoder/block modules to checkpoint."
        )
    if report.get("error"):
        print(
            "warning: native gradient_checkpointing unsupported; "
            f"using ASCR fallback on {report['wrapped_module_count']} modules"
        )
    return report


def train_lumina_lora_smoke(args):
    torch, LoraConfig, TaskType, get_peft_model, AutoTokenizer, model_cls, add_break_line = _load_training_stack(args.repo_path)
    random.seed(int(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    model_kwargs = {"torch_dtype": _dtype_value(torch, args.torch_dtype)}
    if str(args.device_map or "").lower() not in {"", "none", "null"}:
        model_kwargs["device_map"] = args.device_map
    model = model_cls.from_pretrained(args.checkpoint_path, **model_kwargs)
    if "device_map" not in model_kwargs and hasattr(model, "to"):
        model = model.to(device)
    gradient_checkpointing_report = {"requested": bool(args.gradient_checkpointing), "backend": "disabled", "wrapped_module_count": 0}
    if args.gradient_checkpointing:
        gradient_checkpointing_report = _configure_gradient_checkpointing(
            torch,
            model,
            mode=args.gradient_checkpointing_fallback,
        )
    model, lora_parameter_report = _apply_lora_adapter(LoraConfig, TaskType, get_peft_model, model, args)
    model.train()
    rows = list(_jsonl_rows(args.data_jsonl))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError(f"No training rows found in {args.data_jsonl}")
    val_rows = list(_jsonl_rows(args.val_jsonl)) if args.val_jsonl else []
    if args.limit is not None and val_rows:
        val_rows = val_rows[: int(args.limit)]
    optimizer = _build_optimizer(
        torch,
        args.optimizer,
        _trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    losses = []
    epoch_summaries = []
    checkpoints = []
    best_val_loss = None
    best_checkpoint = None
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(int(args.epochs)):
        started_at = time.monotonic()
        random.shuffle(rows)
        total = 0.0
        step_iter, progress_bar = _iter_with_progress(
            enumerate(rows),
            total=len(rows),
            desc=f"epoch {epoch}",
            args=args,
        )
        for step, row in step_iter:
            input_ids, labels = _prepare_lumina_lora_batch(torch, tokenizer, add_break_line, row, args)
            loss = _call_model_loss(torch, model, input_ids, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = float(loss.item())
            total += value
            losses.append({"epoch": epoch, "step": step, "loss": value})
            count = step + 1
            if progress_bar is not None:
                elapsed_s = time.monotonic() - started_at
                progress_bar.set_postfix(
                    loss=f"{value:.4f}",
                    avg_loss=f"{(total / max(1, count)):.4f}",
                    samples_s=f"{(count / max(elapsed_s, 1e-9)):.4f}",
                )
            _log_training_progress(
                epoch,
                step,
                len(rows),
                value,
                total,
                count,
                started_at,
                args,
            )
        if progress_bar is not None:
            progress_bar.close()
        train_loss = total / max(1, len(rows))
        should_validate = bool(val_rows) and _should_validate_epoch(args, epoch, int(args.epochs))
        val_loss = _mean_lora_loss(torch, model, tokenizer, add_break_line, val_rows, args) if should_validate else None
        elapsed_s = time.monotonic() - started_at
        epoch_summaries.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "validated": bool(should_validate),
            "elapsed_s": elapsed_s,
            "samples_per_s": len(rows) / max(elapsed_s, 1e-9),
        })
        suffix = f", val_loss={val_loss:.6f}" if val_loss is not None else ""
        print(
            f"epoch {epoch}: avg_loss={train_loss:.6f}{suffix}, elapsed_s={elapsed_s:.1f}",
            flush=True,
        )
        should_save_epoch = int(args.checkpoint_every_epochs or 0) > 0 and (epoch + 1) % int(args.checkpoint_every_epochs) == 0
        if val_loss is not None:
            improved = best_val_loss is None or val_loss < best_val_loss - float(args.early_stopping_min_delta)
            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_checkpoint = _save_epoch_checkpoint(model, tokenizer, args.output_dir, epoch + 1)
                checkpoints.append(best_checkpoint)
            else:
                epochs_without_improvement += 1
            if int(args.early_stopping_patience or 0) > 0 and epochs_without_improvement >= int(args.early_stopping_patience):
                stopped_early = True
                print(
                    f"early stopping at epoch {epoch}: best_val_loss={best_val_loss:.6f}, "
                    f"patience={int(args.early_stopping_patience)}",
                    flush=True,
                )
                break
        elif should_save_epoch:
            checkpoints.append(_save_epoch_checkpoint(model, tokenizer, args.output_dir, epoch + 1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_checkpoint:
        _copy_checkpoint_to_output(best_checkpoint, output_dir)
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    manifest = {
        "schema_version": "ascr.lumina_lora_smoke.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_jsonl": str(args.data_jsonl),
        "val_jsonl": str(args.val_jsonl) if args.val_jsonl else None,
        "checkpoint_path": str(args.checkpoint_path),
        "repo_path": str(args.repo_path),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "epochs": int(args.epochs),
        "completed_epochs": max((item["epoch"] for item in losses), default=-1) + 1,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "optimizer": str(args.optimizer),
        "image_size": int(args.image_size),
        "max_seq_len": int(args.max_seq_len),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "answer_mask_mode": str(args.answer_mask_mode),
        "ignore_pad_labels": bool(args.ignore_pad_labels),
        "torch_dtype": str(args.torch_dtype),
        "device_map": str(args.device_map),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "gradient_checkpointing_report": gradient_checkpointing_report,
        "cache_report": _lumina_lora_cache_report(),
        "lora_parameter_report": lora_parameter_report,
        "resume_from_adapter": str(args.resume_from_adapter) if args.resume_from_adapter else None,
        "checkpoint_every_epochs": int(args.checkpoint_every_epochs or 0),
        "early_stopping_patience": int(args.early_stopping_patience or 0),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "validation_every_epochs": _validation_every_epochs(args),
        "progress_bar": bool(getattr(args, "progress_bar", True)),
        "progress_every_steps": _progress_every_steps(args),
        "stopped_early": stopped_early,
        "best_val_loss": best_val_loss,
        "best_checkpoint": best_checkpoint,
        "checkpoints": checkpoints,
        "device": device,
        "epoch_summaries": epoch_summaries,
        "losses": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Run a single-GPU Lumina-DiMOO LoRA SFT smoke on ASCR SemanticEvaluation JSON.")
    parser.add_argument("--repo-path", default=os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO"))
    parser.add_argument("--checkpoint-path", default=os.environ.get("LUMINA_MODEL_PATH", "models/lumina-dimoo"))
    parser.add_argument("--data-jsonl", default="outputs/stage2_lumina_native/lumina_sft_data_v2/train.jsonl")
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--output-dir", default="outputs/stage2_lumina_native/lora_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prompt-max-length", type=int, default=512)
    parser.add_argument("--answer-max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--resume-from-adapter", default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs and always on the final epoch; use 0 to disable validation.",
    )
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=25,
        help="Print a flushed plain-text training progress line every N steps; use 0 to disable.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars when tqdm is installed.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Model device_map for single-process training. Use 'none' under torchrun/DDP.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gradient-checkpointing-fallback",
        choices=["auto", "off", "force"],
        default="auto",
        help="Fallback when the Lumina model class does not advertise HuggingFace gradient checkpointing support.",
    )
    parser.add_argument(
        "--answer-mask-mode",
        choices=["random", "all"],
        default="random",
        help="Masking strategy for answer tokens. Use 'all' to match Lumina answer generation more closely.",
    )
    parser.add_argument(
        "--ignore-pad-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not train loss on synthetic answer padding slots.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    train_lumina_lora_smoke(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

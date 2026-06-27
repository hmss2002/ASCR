import argparse
from datetime import datetime, timezone
import json
import math
import os
import pickle
from pathlib import Path
import random


SP = {
    "mask": 126336,
    "newline": 126084,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,
    "eoi": 126350,
    "padding": 126339,
}


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
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer
    from model import LLaDAForMultiModalGeneration
    from utils.image_utils import add_break_line

    return torch, LoraConfig, TaskType, get_peft_model, AutoTokenizer, LLaDAForMultiModalGeneration, add_break_line


def train_lumina_lora_smoke(args):
    torch, LoraConfig, TaskType, get_peft_model, AutoTokenizer, model_cls, add_break_line = _load_training_stack(args.repo_path)
    random.seed(int(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    dtype_name = str(args.torch_dtype).lower()
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
        raise ValueError(f"Unsupported torch_dtype: {args.torch_dtype}")
    model = model_cls.from_pretrained(args.checkpoint_path, torch_dtype=dtype_map[dtype_name], device_map="auto")
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            print("warning: model does not expose gradient_checkpointing_enable(); continuing without it")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=[part.strip() for part in str(args.target_modules).split(",") if part.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    rows = list(_jsonl_rows(args.data_jsonl))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError(f"No training rows found in {args.data_jsonl}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    losses = []
    for epoch in range(int(args.epochs)):
        random.shuffle(rows)
        total = 0.0
        for step, row in enumerate(rows):
            with open(row["user_image"], "rb") as handle:
                image_payload = pickle.load(handle)
            image_tokens, token_h, token_w = _center_crop_tokens(
                image_payload["input_ids"],
                image_payload["height"],
                image_payload["width"],
                args.image_size,
            )
            image_tokens = add_break_line(image_tokens, token_h, token_w, new_number=SP["newline"])
            instruction = "<system>" + row["system_prompt"] + "</system>" + "<user>" + row["user_prompt"] + "</user>"
            inst_ids = tokenizer(
                instruction,
                truncation=True,
                max_length=int(args.prompt_max_length),
                padding=False,
                return_tensors="pt",
            ).input_ids[0].tolist()
            inst_ids = inst_ids[:-1] + [SP["boi"]] + image_tokens + [SP["eoi"]] + inst_ids[-1:]
            answer_ids = tokenizer(
                row["answer_text"] + "</answer>",
                truncation=True,
                max_length=int(args.answer_max_length),
                padding=False,
                return_tensors="pt",
            ).input_ids[0].tolist()
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
            loss = model(input_ids=[input_ids], labels=[labels])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = float(loss.item())
            total += value
            losses.append({"epoch": epoch, "step": step, "loss": value})
        print(f"epoch {epoch}: avg_loss={total / max(1, len(rows)):.6f}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest = {
        "schema_version": "ascr.lumina_lora_smoke.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_jsonl": str(args.data_jsonl),
        "checkpoint_path": str(args.checkpoint_path),
        "repo_path": str(args.repo_path),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "image_size": int(args.image_size),
        "max_seq_len": int(args.max_seq_len),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "answer_mask_mode": str(args.answer_mask_mode),
        "ignore_pad_labels": bool(args.ignore_pad_labels),
        "torch_dtype": str(args.torch_dtype),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "device": device,
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
    parser.add_argument("--output-dir", default="outputs/stage2_lumina_native/lora_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prompt-max-length", type=int, default=512)
    parser.add_argument("--answer-max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
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

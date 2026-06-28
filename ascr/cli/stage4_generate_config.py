"""Generate Stage-4 MMU LoRA config files from grid/profile choices."""

from __future__ import annotations

import argparse
from pathlib import Path


def _dump_yaml(path, payload):
    lines = []
    for key, value in payload.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif value is None:
            text = "null"
        else:
            text = str(value)
        lines.append(f"{key}: {text}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_config(grid, profile="l40s_1024_gc", prompt_variant="schema_example", kind="train"):
    grid = int(grid)
    base_dir = f"outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid{grid}/vq_tokens"
    if kind == "sft":
        return {
            "dataset": "outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl",
            "output_dir": f"{base_dir}/sft",
            "grid_size": grid,
            "max_selected_cells": grid,
            "eval_mode": "holdout",
            "train_ratio": 0.75,
            "seed": 0,
            "limit": None,
            "project_root": ".",
            "input_mode": "vq_tokens",
            "target_schema": "localization_cells",
            "prompt_variant": prompt_variant,
        }
    if kind == "probe":
        suffix = "l40s_1024px_gc" if profile == "l40s_1024_gc" else "l40s"
        adapter = f"lora_{suffix}_adam8bit" if profile == "l40s_1024_gc" else "lora_l40s"
        return {
            "dataset": "outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl",
            "output_dir": f"{base_dir}/probe_lora_{suffix}_eval",
            "grid_size": grid,
            "max_selected_cells": grid,
            "top_k": min(4, max(2, grid // 2)),
            "limit": None,
            "split_manifest": f"{base_dir}/sft/split_manifest.json",
            "split": "eval",
            "input_mode": "vq_tokens",
            "use_vq_tokens": True,
            "target_schema": "localization_cells",
            "prompt_variant": prompt_variant,
            "lora_path": f"{base_dir}/{adapter}",
            "repo_path": "third_party/Lumina-DiMOO",
            "checkpoint_path": "models/lumina-dimoo",
            "device": "cuda",
            "image_size": 1024,
            "max_new_tokens": 384,
            "answer_steps": 64,
            "answer_block_length": 128,
            "answer_temperature": 0.0,
            "answer_cfg_scale": 0.0,
        }
    if profile != "l40s_1024_gc":
        raise ValueError(f"Unsupported generated train profile: {profile}")
    return {
        "repo_path": "third_party/Lumina-DiMOO",
        "checkpoint_path": "models/lumina-dimoo",
        "data_jsonl": f"{base_dir}/lumina_sft/train.jsonl",
        "output_dir": f"{base_dir}/lora_l40s_1024px_gc_adam8bit",
        "epochs": 15,
        "limit": None,
        "lr": "3.0e-5",
        "weight_decay": 0.01,
        "optimizer": "adamw8bit",
        "image_size": 1024,
        "max_seq_len": 6144,
        "prompt_max_length": 512,
        "answer_max_length": 512,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,
        "gradient_checkpointing_fallback": "force",
        "answer_mask_mode": "all",
        "ignore_pad_labels": True,
        "seed": 0,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Stage-4 MMU LoRA YAML configs.")
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--profile", default="l40s_1024_gc")
    parser.add_argument("--prompt-variant", default="schema_example")
    parser.add_argument("--kind", choices=["train", "probe", "sft"], default="train")
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = _dump_yaml(args.output, build_config(args.grid, profile=args.profile, prompt_variant=args.prompt_variant, kind=args.kind))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


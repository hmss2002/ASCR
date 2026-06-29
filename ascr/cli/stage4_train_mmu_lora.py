import argparse
import json

from ascr.core.config import load_config
from ascr.training.train_lumina_lora_smoke import build_parser as build_lora_parser
from ascr.training.train_lumina_lora_smoke import train_lumina_lora_smoke


def build_parser():
    parser = argparse.ArgumentParser(description="Train Stage-4 Lumina MMU LoRA from prepared localization data.")
    parser.add_argument("--config", default="configs/stage4/self_corrupt/mmu_lora_train_hard64.yaml")
    parser.add_argument("--data-jsonl", default=None)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--resume-from-adapter", default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing-fallback", choices=["auto", "off", "force"], default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
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
        value = getattr(args, key)
        if value is not None:
            setattr(lora_args, key, value)
    manifest = train_lumina_lora_smoke(lora_args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

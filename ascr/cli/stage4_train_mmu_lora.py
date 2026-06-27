import argparse
import json

from ascr.core.config import load_config
from ascr.training.train_lumina_lora_smoke import build_parser as build_lora_parser
from ascr.training.train_lumina_lora_smoke import train_lumina_lora_smoke


def build_parser():
    parser = argparse.ArgumentParser(description="Train Stage-4 Lumina MMU LoRA from prepared localization data.")
    parser.add_argument("--config", default="configs/stage4/self_corrupt/mmu_lora_train_hard64.yaml")
    parser.add_argument("--data-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    lora_args = build_lora_parser().parse_args([])
    for key, value in config.items():
        setattr(lora_args, key.replace("-", "_"), value)
    if args.data_jsonl:
        lora_args.data_jsonl = args.data_jsonl
    if args.output_dir:
        lora_args.output_dir = args.output_dir
    manifest = train_lumina_lora_smoke(lora_args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

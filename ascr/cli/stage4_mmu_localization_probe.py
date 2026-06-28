import argparse
import json
import os

from ascr.core.config import load_config
from ascr.training.stage4_mmu_lora import (
    PROMPT_VARIANT_CHOICES,
    PROMPT_VARIANT_DEFAULT,
    run_mmu_localization_probe,
)


def _read_sample_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Probe Lumina MMU self-corruption localization with optional LoRA adapter.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage4_self_corrupt/mmu_lora_hard64/probe_zero")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-selected-cells", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-ids-file", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--input-mode", choices=["vq_tokens", "decoded_image"], default=None)
    parser.add_argument("--use-vq-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-schema", choices=["localization_cells", "semantic_evaluation"], default="localization_cells")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANT_CHOICES, default=PROMPT_VARIANT_DEFAULT)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--repo-path", default=os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO"))
    parser.add_argument("--checkpoint-path", default=os.environ.get("LUMINA_MODEL_PATH", "models/lumina-dimoo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--answer-steps", type=int, default=64)
    parser.add_argument("--answer-block-length", type=int, default=128)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--answer-cfg-scale", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    summary = run_mmu_localization_probe(
        config.get("dataset", args.dataset),
        config.get("output_dir", args.output_dir),
        grid_size=int(config.get("grid_size", args.grid_size)),
        max_selected_cells=int(config.get("max_selected_cells", args.max_selected_cells)),
        top_k=int(config.get("top_k", args.top_k)),
        limit=config.get("limit", args.limit),
        sample_offset=int(config.get("sample_offset", args.sample_offset) or 0),
        sample_ids=config.get("sample_ids", _read_sample_ids(args.sample_ids_file)),
        split_manifest=config.get("split_manifest", args.split_manifest),
        split=config.get("split", args.split),
        input_mode=config.get("input_mode", args.input_mode),
        use_vq_tokens=bool(config.get("use_vq_tokens", args.use_vq_tokens)),
        target_schema=config.get("target_schema", args.target_schema),
        prompt_variant=config.get("prompt_variant", args.prompt_variant),
        lora_path=config.get("lora_path", args.lora_path),
        repo_path=config.get("repo_path", args.repo_path),
        checkpoint_path=config.get("checkpoint_path", args.checkpoint_path),
        device=config.get("device", args.device),
        image_size=int(config.get("image_size", args.image_size)),
        max_new_tokens=int(config.get("max_new_tokens", args.max_new_tokens)),
        answer_steps=int(config.get("answer_steps", args.answer_steps)),
        answer_block_length=int(config.get("answer_block_length", args.answer_block_length)),
        answer_temperature=float(config.get("answer_temperature", args.answer_temperature)),
        answer_cfg_scale=float(config.get("answer_cfg_scale", args.answer_cfg_scale)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

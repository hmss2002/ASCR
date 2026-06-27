import argparse
import json

from ascr.core.config import load_config
from ascr.training.stage4_mmu_lora import prepare_mmu_sft_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare Stage-4 Lumina MMU LoRA SFT data from a self-corruption dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage4_self_corrupt/mmu_lora_hard64/sft")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-selected-cells", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-mode", choices=["holdout", "resubstitution"], default="holdout")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input-mode", choices=["vq_tokens", "decoded_image", "both"], default="vq_tokens")
    parser.add_argument("--target-schema", choices=["localization_cells", "semantic_evaluation"], default="localization_cells")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    manifest = prepare_mmu_sft_dataset(
        config.get("dataset", args.dataset),
        config.get("output_dir", args.output_dir),
        grid_size=int(config.get("grid_size", args.grid_size)),
        max_selected_cells=int(config.get("max_selected_cells", args.max_selected_cells)),
        train_ratio=float(config.get("train_ratio", args.train_ratio)),
        seed=int(config.get("seed", args.seed)),
        eval_mode=config.get("eval_mode", args.eval_mode),
        limit=config.get("limit", args.limit),
        project_root=config.get("project_root", args.project_root),
        input_mode=config.get("input_mode", args.input_mode),
        target_schema=config.get("target_schema", args.target_schema),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

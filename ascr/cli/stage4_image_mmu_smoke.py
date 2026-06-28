import argparse
import json
import os

from ascr.training.stage4_mmu_lora import run_mmu_localization_probe


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke-test Lumina answer_image() on Stage-4 self-corruption localization rows.")
    parser.add_argument("--dataset", default="outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage4_self_corrupt/image_mmu_smoke")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-selected-cells", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=0)
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
    summary = run_mmu_localization_probe(
        args.dataset,
        args.output_dir,
        grid_size=args.grid_size,
        max_selected_cells=args.max_selected_cells,
        top_k=args.top_k,
        limit=args.limit,
        sample_offset=args.sample_offset,
        input_mode="decoded_image",
        repo_path=args.repo_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        image_size=args.image_size,
        max_new_tokens=args.max_new_tokens,
        answer_steps=args.answer_steps,
        answer_block_length=args.answer_block_length,
        answer_temperature=args.answer_temperature,
        answer_cfg_scale=args.answer_cfg_scale,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

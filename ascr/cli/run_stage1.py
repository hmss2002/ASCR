import argparse
import json
from pathlib import Path

from ascr.core.config import load_config
from ascr.core.loop import ASCRLoop, run_config_from_mapping
from ascr.evaluators.registry import build_evaluator
from ascr.generators.registry import build_generator
from ascr.revision.selector import GridSemanticReopeningSelector


def build_parser():
    parser = argparse.ArgumentParser(description="Run ASCR Stage 1 zero-training loop.")
    parser.add_argument("--config", default=None, help="Path to YAML or JSON config.")
    parser.add_argument("--prompt", default="A red cube left of a blue sphere", help="Original text-to-image prompt.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max ASCR iterations.")
    parser.add_argument("--generator", default=None, choices=["mock", "showo"], help="Generator adapter.")
    parser.add_argument("--evaluator", default=None, choices=["mock", "local_vlm", "local-vlm", "showo_mmu", "showo-mmu", "showo_vlm", "showo-vlm", "qwen_vl", "qwen-vl", "qwen3_6", "qwen36"], help="Semantic evaluator adapter.")
    parser.add_argument("--dry-run", action="store_true", help="Force mock generator and mock evaluator.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.dry_run:
        generator_name = "mock"
        evaluator_name = "mock"
    else:
        generator_name = args.generator or config.get("generator", {}).get("name", "mock")
        evaluator_name = args.evaluator or config.get("evaluator", {}).get("name", "mock")
    generator_config = dict(config)
    generator_config["token_grid_size"] = int(config.get("token_grid_size", 16))
    generator_config["image_size"] = int(config.get("image_size", 256))
    generator = build_generator(generator_name, generator_config)
    evaluator = build_evaluator(evaluator_name, config)
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get("coarse_grid_size", 4)),
        token_grid_size=int(config.get("token_grid_size", 16)),
        dilation=int(config.get("dilation", 1)),
    )
    run_config = run_config_from_mapping(config)
    loop = ASCRLoop(generator, evaluator, selector, run_config)
    summary = loop.run(args.prompt, project_root=Path.cwd())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

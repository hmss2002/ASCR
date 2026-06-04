import argparse
import json
from pathlib import Path

from ascr.core.config import load_config
from ascr.core.loop import run_config_from_mapping
from ascr.core.loop_direct import DirectTokenReopenLoop
from ascr.evaluators.registry import build_evaluator
from ascr.generators.registry import build_generator
from ascr.revision.selector import DirectTokenReopeningSelector


def build_parser():
    parser = argparse.ArgumentParser(description="Run ASCR Stage 1 direct-token reopen loop (no coarse grid / dilation / upsampling).")
    parser.add_argument("--config", default=None, help="Path to YAML or JSON config.")
    parser.add_argument("--prompt", default="A red cube left of a blue sphere", help="Original text-to-image prompt.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max ASCR iterations.")
    parser.add_argument("--generator", default=None, choices=["mock", "showo"], help="Generator adapter.")
    parser.add_argument("--evaluator", default=None, help="Semantic evaluator adapter (e.g. qwen_vl_token, mock).")
    parser.add_argument("--select-grid-size", type=int, default=None, help="Token-selection granularity (default token_grid_size).")
    parser.add_argument("--dry-run", action="store_true", help="Force mock generator and mock evaluator.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.select_grid_size is not None:
        config["select_grid_size"] = args.select_grid_size
        config.setdefault("evaluator", {})["select_grid_size"] = args.select_grid_size
        config.setdefault("selector", {})["select_grid_size"] = args.select_grid_size
    if args.dry_run:
        generator_name = "mock"
        evaluator_name = "mock"
    else:
        generator_name = args.generator or config.get("generator", {}).get("name", "mock")
        evaluator_name = args.evaluator or config.get("evaluator", {}).get("name", "qwen_vl_token")
    token_grid_size = int(config.get("token_grid_size", 32))
    generator_config = dict(config)
    generator_config["token_grid_size"] = token_grid_size
    generator_config["image_size"] = int(config.get("image_size", 512))
    generator = build_generator(generator_name, generator_config)
    evaluator = build_evaluator(evaluator_name, config)
    selector = DirectTokenReopeningSelector(
        token_grid_size=token_grid_size,
        select_grid_size=int(config.get("select_grid_size", config.get("selector", {}).get("select_grid_size", token_grid_size))),
        dilation=int(config.get("dilation", config.get("selector", {}).get("dilation", 0))),
    )
    run_config = run_config_from_mapping(config)
    loop = DirectTokenReopenLoop(generator, evaluator, selector, run_config, label_step=int(config.get("label_step", 4)))
    summary = loop.run(args.prompt, project_root=Path.cwd())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

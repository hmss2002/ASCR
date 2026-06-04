import argparse
import json
from pathlib import Path

from ascr.core.config import load_config
from ascr.core.loop import ASCRLoop, run_config_from_mapping
from ascr.evaluators.registry import build_evaluator
from ascr.generators.registry import build_generator
from ascr.revision.selector import GridSemanticReopeningSelector


def build_parser():
    parser = argparse.ArgumentParser(description="Run ASCR Stage 1 on MMaDA-8B with MMaDA itself as a COARSE (4x4) self-selector following the original ASCR coarse-then-dilate strategy (single shared 8B model).")
    parser.add_argument("--config", default="configs/stage1_mmada8b_self_coarse.yaml", help="Path to YAML or JSON config.")
    parser.add_argument("--prompt", default="A red cube left of a blue sphere", help="Original text-to-image prompt.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max ASCR iterations.")
    parser.add_argument("--dry-run", action="store_true", help="Force mock generator and mock evaluator (no 8B load).")
    return parser


def _attach_shared_engine(generator, evaluator):
    """Make the MMaDA coarse self-evaluator reuse the generator's loaded 8B engine."""
    use = getattr(evaluator, "use_engine", None) or getattr(evaluator, "attach_engine", None)
    get_engine = getattr(generator, "_engine", None)
    if callable(use) and callable(get_engine):
        return bool(use(get_engine()))
    return False


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
        generator_name = config.get("generator", {}).get("name", "mmada")
        evaluator_name = config.get("evaluator", {}).get("name", "mmada_self_coarse")
    token_grid_size = int(config.get("token_grid_size", 32))
    generator_config = dict(config)
    generator_config["token_grid_size"] = token_grid_size
    generator_config["image_size"] = int(config.get("image_size", 512))
    generator = build_generator(generator_name, generator_config)
    evaluator = build_evaluator(evaluator_name, config)
    shared = _attach_shared_engine(generator, evaluator)
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get("coarse_grid_size", config.get("selector", {}).get("coarse_grid_size", 4))),
        token_grid_size=token_grid_size,
        dilation=int(config.get("dilation", config.get("selector", {}).get("dilation", 1))),
    )
    run_config = run_config_from_mapping(config)
    loop = ASCRLoop(generator, evaluator, selector, run_config)
    summary = loop.run(args.prompt, project_root=Path.cwd())
    summary["shared_engine"] = shared
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

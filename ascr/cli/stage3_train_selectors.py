import argparse
import json
from pathlib import Path

from ascr.core.config import load_config
from ascr.training.stage3_selectors import DEFAULT_BASELINES, train_selector_suite


def _config_list(config, key, default):
    value = config.get(key, default)
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate Stage-3 self-corruption selector baselines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--grid-size", action="append", type=int, default=None)
    parser.add_argument("--baseline", action="append", choices=DEFAULT_BASELINES, default=None)
    parser.add_argument("--eval-mode", choices=["resubstitution", "holdout"], default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--prompt-hash-dims", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--l2", type=float, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    dataset = args.dataset or config.get("dataset") or "outputs/stage3_self_corrupt/datasets/locality_smoke_v1/dataset.jsonl"
    output_dir = args.output_dir or config.get("output_dir") or "outputs/stage3_self_corrupt/selectors/locality_smoke_v1"
    grid_sizes = args.grid_size or [int(value) for value in _config_list(config, "grid_sizes", [4, 8, 16])]
    baselines = args.baseline or _config_list(config, "baselines", DEFAULT_BASELINES)
    summary = train_selector_suite(
        dataset,
        output_dir,
        grid_sizes=grid_sizes,
        baselines=baselines,
        eval_mode=args.eval_mode or config.get("eval_mode", "holdout"),
        train_ratio=float(args.train_ratio if args.train_ratio is not None else config.get("train_ratio", 0.75)),
        seed=int(args.seed if args.seed is not None else config.get("seed", 0)),
        top_k=args.top_k if args.top_k is not None else config.get("top_k"),
        project_root=args.project_root or config.get("project_root") or Path.cwd(),
        prompt_hash_dims=int(args.prompt_hash_dims if args.prompt_hash_dims is not None else config.get("prompt_hash_dims", 16)),
        epochs=int(args.epochs if args.epochs is not None else config.get("epochs", 120)),
        learning_rate=float(args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 0.08)),
        l2=float(args.l2 if args.l2 is not None else config.get("l2", 0.001)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

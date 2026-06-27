import argparse
import json

from ascr.core.config import load_config
from ascr.training.stage4_repair import train_repair_head


def build_parser():
    parser = argparse.ArgumentParser(description="Train a lightweight Phase-4 repair head from extracted Lumina hidden features.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--features", default="outputs/stage4_self_corrupt/hidden_features_hard64_grid16/hidden_features.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage4_self_corrupt/repair_head_hard64_grid16")
    parser.add_argument("--eval-mode", choices=["resubstitution", "holdout"], default="holdout")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    result = train_repair_head(
        config.get("features", args.features),
        config.get("output_dir", args.output_dir),
        eval_mode=config.get("eval_mode", args.eval_mode),
        train_ratio=float(config.get("train_ratio", args.train_ratio)),
        seed=int(config.get("seed", args.seed)),
        top_k=int(config.get("top_k", args.top_k)),
        epochs=int(config.get("epochs", args.epochs)),
        learning_rate=float(config.get("learning_rate", args.learning_rate)),
        l2=float(config.get("l2", args.l2)),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import os

from ascr.core.config import load_config
from ascr.training.stage4_repair import probe_hidden_state_support


def build_parser():
    parser = argparse.ArgumentParser(description="Probe whether Lumina forward exposes hidden states for Phase-4 repair heads.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="outputs/stage3_self_corrupt/datasets/locality_hard64_v1/dataset.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage4_self_corrupt/hidden_probe_hard64")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--repo-path", default=os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO"))
    parser.add_argument("--checkpoint-path", default=os.environ.get("LUMINA_MODEL_PATH", "models/lumina-dimoo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--token-grid-size", type=int, default=64)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    report = probe_hidden_state_support(
        config.get("dataset", args.dataset),
        config.get("output_dir", args.output_dir),
        limit=config.get("limit", args.limit),
        repo_path=config.get("repo_path", args.repo_path),
        checkpoint_path=config.get("checkpoint_path", args.checkpoint_path),
        device=config.get("device", args.device),
        image_size=int(config.get("image_size", args.image_size)),
        token_grid_size=int(config.get("token_grid_size", args.token_grid_size)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

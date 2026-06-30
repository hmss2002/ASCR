import argparse
import json

from ascr.analysis.stage3_token_repair import build_token_repair_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="Build the canonical token-only Stage-3 8x8 repair dataset.")
    parser.add_argument("--clean-manifest", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-rows", type=int, default=30000)
    parser.add_argument("--negative-rows", type=int, default=10000)
    parser.add_argument("--variants-per-clean", type=int, default=3)
    parser.add_argument("--mask-sizes", nargs="+", default=["1", "2", "4", "8"])
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["random_replace", "local_shuffle", "neighbor_copy", "transplant"],
    )
    parser.add_argument("--action-grid-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project-root", default=".")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = build_token_repair_dataset(
        args.clean_manifest,
        args.output_dir,
        positive_rows=args.positive_rows,
        negative_rows=args.negative_rows,
        variants_per_clean=args.variants_per_clean,
        mask_sizes=args.mask_sizes,
        operators=args.operators,
        action_grid_size=args.action_grid_size,
        seed=args.seed,
        project_root=args.project_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

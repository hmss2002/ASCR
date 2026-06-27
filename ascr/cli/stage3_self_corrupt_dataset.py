import argparse
import json

from ascr.analysis.stage3_self_corrupt import build_self_corrupt_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="Build a Phase-2 self-corruption dataset from a Stage-3 locality probe manifest.")
    parser.add_argument("--manifest", required=True, help="Path to locality probe manifest.jsonl.")
    parser.add_argument("--summary", default=None, help="Optional path to locality probe summary.json.")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset.jsonl and dataset_manifest.json.")
    parser.add_argument("--project-root", default=".", help="Project root for normalising absolute paths.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = build_self_corrupt_dataset(
        args.manifest,
        args.output_dir,
        summary_path=args.summary,
        project_root=args.project_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

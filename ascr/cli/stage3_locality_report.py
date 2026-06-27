import argparse
import json
from pathlib import Path

from ascr.analysis.stage3_self_corrupt import build_locality_report, format_locality_markdown, write_json


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate Stage-3 self-corruption locality probe metrics.")
    parser.add_argument("--manifest", required=True, help="Path to locality probe manifest.jsonl.")
    parser.add_argument("--summary", default=None, help="Optional path to locality probe summary.json.")
    parser.add_argument("--output-dir", default=None, help="Directory for locality_report.json and locality_report.md.")
    parser.add_argument("--json-output", default=None, help="Override JSON report output path.")
    parser.add_argument("--markdown-output", default=None, help="Override Markdown report output path.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "report"
    json_output = Path(args.json_output) if args.json_output else output_dir / "locality_report.json"
    markdown_output = Path(args.markdown_output) if args.markdown_output else output_dir / "locality_report.md"
    report = build_locality_report(manifest_path, summary_path=args.summary)
    write_json(json_output, report)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(format_locality_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

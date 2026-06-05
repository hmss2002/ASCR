import argparse
import json
from pathlib import Path

from ascr.cli.compare_showo_ascr import build_suite, suite_to_markdown


def load_results(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "results" in data:
        results = data["results"]
    else:
        results = [data]
    for result in results:
        result.setdefault("source_suite_path", str(path))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate Show-o baseline-vs-ASCR comparison suites.")
    parser.add_argument("inputs", nargs="+", help="suite.json or comparison.json files to aggregate")
    parser.add_argument("--output", required=True, help="Output aggregate suite.json path")
    parser.add_argument("--metadata-json", default=None, help="Optional JSON object to store under suite sharded metadata")
    parser.add_argument("--metadata-file", default=None, help="Optional JSON file to store under suite sharded metadata")
    args = parser.parse_args(argv)

    results = []
    for input_path in args.inputs:
        results.extend(load_results(input_path))

    suite = build_suite(results)
    if args.metadata_file:
        suite["sharded"] = json.loads(Path(args.metadata_file).read_text(encoding="utf-8"))
    elif args.metadata_json:
        suite["sharded"] = json.loads(args.metadata_json)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(suite_to_markdown(suite), encoding="utf-8")
    print(json.dumps({
        "suite_path": str(output_path),
        "suite_markdown_path": str(output_path.with_suffix(".md")),
        "prompt_count": suite["prompt_count"],
        "total_ascr_insertions": suite["total_ascr_insertions"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

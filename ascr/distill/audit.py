import argparse
from collections import Counter
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            records.append({"_parse_error": str(exc), "_line": line_no})
    return records


def score_bucket(value):
    try:
        score = float(value)
    except Exception:
        return "invalid"
    if score < 0.25:
        return "0.00-0.24"
    if score < 0.5:
        return "0.25-0.49"
    if score < 0.75:
        return "0.50-0.74"
    if score < 1.0:
        return "0.75-0.99"
    return "1.00"


def audit_distill_dir(distill_dir):
    root = Path(distill_dir)
    quality = read_jsonl(root / "quality_labels.jsonl")
    localization = read_jsonl(root / "localization_labels.jsonl")
    errors = read_jsonl(root / "errors.jsonl")
    winner_counts = Counter()
    baseline_buckets = Counter()
    final_buckets = Counter()
    quality_parse_errors = 0
    for record in quality:
        if record.get("_parse_error"):
            quality_parse_errors += 1
            continue
        payload = record.get("quality", {})
        winner_counts[str(payload.get("winner", "missing"))] += 1
        baseline_buckets[score_bucket(payload.get("baseline_score"))] += 1
        final_buckets[score_bucket(payload.get("final_score"))] += 1
    localization_parse_errors = 0
    has_error_counts = Counter()
    selected_cell_counts = Counter()
    missing_path_count = 0
    for record in localization:
        if record.get("_parse_error"):
            localization_parse_errors += 1
            continue
        evaluation = record.get("evaluation", {})
        has_error_counts[str(bool(evaluation.get("has_error")))] += 1
        selected = 0
        for region in evaluation.get("regions", []) or []:
            selected += len(region.get("cells", []) or [])
        selected_cell_counts[str(selected)] += 1
        for key in ("grid_image", "trace_path", "record_path"):
            if not record.get(key):
                missing_path_count += 1
    error_by_kind = Counter(str(record.get("kind", "unknown")) for record in errors if not record.get("_parse_error"))
    report = {
        "distill_dir": str(root),
        "counts": {
            "quality": len(quality),
            "localization": len(localization),
            "errors": len(errors),
            "quality_parse_errors": quality_parse_errors,
            "localization_parse_errors": localization_parse_errors,
            "missing_path_fields": missing_path_count,
        },
        "quality": {
            "winner_counts": dict(sorted(winner_counts.items())),
            "baseline_score_buckets": dict(sorted(baseline_buckets.items())),
            "final_score_buckets": dict(sorted(final_buckets.items())),
        },
        "localization": {
            "has_error_counts": dict(sorted(has_error_counts.items())),
            "selected_cell_counts": dict(sorted(selected_cell_counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999)),
        },
        "errors": {
            "by_kind": dict(sorted(error_by_kind.items())),
        },
    }
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Audit ASCR API teacher distillation outputs.")
    parser.add_argument("--distill-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = audit_distill_dir(args.distill_dir)
    output = Path(args.output) if args.output else Path(args.distill_dir) / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["counts"]["quality_parse_errors"] == 0 and report["counts"]["localization_parse_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_delta(before, after, key):
    left = before.get(key)
    right = after.get(key)
    if left is None or right is None:
        return None
    return right - left


def compare_summaries(baseline_summary, candidate_summary, output):
    baseline = load_json(baseline_summary)
    candidate = load_json(candidate_summary)
    report = {
        "schema_version": "ascr.image_judge_comparison.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "baseline_summary": str(baseline_summary),
        "candidate_summary": str(candidate_summary),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "row_count": score_delta(baseline, candidate, "row_count"),
            "error_count": score_delta(baseline, candidate, "error_count"),
            "mean_before_score": score_delta(baseline, candidate, "mean_before_score"),
            "mean_after_score": score_delta(baseline, candidate, "mean_after_score"),
            "mean_delta_after_minus_before": score_delta(baseline, candidate, "mean_delta_after_minus_before"),
            "winner_after": int(candidate.get("winners", {}).get("after", 0)) - int(baseline.get("winners", {}).get("after", 0)),
            "winner_before": int(candidate.get("winners", {}).get("before", 0)) - int(baseline.get("winners", {}).get("before", 0)),
            "winner_tie": int(candidate.get("winners", {}).get("tie", 0)) - int(baseline.get("winners", {}).get("tie", 0)),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Compare two ASCR before/after API judge summaries.")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = compare_summaries(args.baseline_summary, args.candidate_summary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

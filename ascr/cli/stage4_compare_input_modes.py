import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


HIGHER_IS_BETTER = {
    "parse_rate",
    "hit_any_rate",
    "exact_match_rate",
    "mean_precision_at_k",
    "mean_recall_at_k",
    "mean_f1_at_k",
    "mean_iou",
}

LOWER_IS_BETTER = {
    "malformed_count",
    "call_error_count",
    "mean_distance_to_target_cells",
    "mean_latency_ms",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_value(summary, metric):
    if metric in summary:
        return summary.get(metric)
    return (summary.get("metrics") or {}).get(metric)


def _winner(values, metric):
    numeric = [
        (label, value)
        for label, value in values.items()
        if isinstance(value, (int, float)) and value is not None
    ]
    if len(numeric) < 2:
        return None
    if metric in LOWER_IS_BETTER:
        best = min(value for _label, value in numeric)
    else:
        best = max(value for _label, value in numeric)
    winners = [label for label, value in numeric if value == best]
    return "tie" if len(winners) > 1 else winners[0]


def compare_probe_summaries(probe_paths, labels=None):
    probe_paths = [Path(path) for path in probe_paths]
    labels = list(labels or [])
    if labels and len(labels) != len(probe_paths):
        raise ValueError("--labels must have the same length as --probes")
    summaries = []
    for index, path in enumerate(probe_paths):
        summary = _read_json(path)
        label = labels[index] if labels else summary.get("input_mode") or path.parent.name
        summaries.append({"label": label, "path": str(path), "summary": summary})

    metrics = [
        "row_count",
        "parse_rate",
        "parsed_count",
        "malformed_count",
        "call_error_count",
        "mean_latency_ms",
        "hit_any_rate",
        "exact_match_rate",
        "mean_precision_at_k",
        "mean_recall_at_k",
        "mean_f1_at_k",
        "mean_iou",
        "mean_distance_to_target_cells",
        "mean_selected_count",
    ]
    rows = []
    for metric in metrics:
        values = {
            item["label"]: _metric_value(item["summary"], metric)
            for item in summaries
        }
        rows.append({
            "metric": metric,
            "values": values,
            "winner": _winner(values, metric),
        })
    return {
        "schema_version": "ascr.stage4.input_mode_comparison.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "probes": summaries,
        "metrics": rows,
    }


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_comparison(output_dir, comparison):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    labels = [item["label"] for item in comparison["probes"]]
    header = ["Metric"] + labels + ["Winner"]
    lines = [
        "# Stage-4 Input-Mode Comparison",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in comparison["metrics"]:
        values = [_format_value(row["values"].get(label)) for label in labels]
        lines.append("| " + " | ".join([row["metric"]] + values + [row.get("winner") or ""]) + " |")
    lines.append("")
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "comparison_json": str(output_dir / "comparison.json"),
        "comparison_md": str(output_dir / "comparison.md"),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Compare Stage-4 MMU localization probe summaries across input modes.")
    parser.add_argument("--probes", nargs="+", default=None, help="Probe summary.json paths.")
    parser.add_argument("--labels", nargs="+", default=None, help="Display labels matching --probes.")
    parser.add_argument("--vq-tokens-probe", default=None, help="Convenience path for the VQ-token summary.json.")
    parser.add_argument("--decoded-image-probe", default=None, help="Convenience path for the decoded-image summary.json.")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    probes = list(args.probes or [])
    labels = args.labels
    if args.vq_tokens_probe or args.decoded_image_probe:
        probes = []
        labels = []
        if args.vq_tokens_probe:
            probes.append(args.vq_tokens_probe)
            labels.append("vq_tokens")
        if args.decoded_image_probe:
            probes.append(args.decoded_image_probe)
            labels.append("decoded_image")
    if len(probes) < 2:
        raise SystemExit("Provide at least two probe summary paths.")
    comparison = compare_probe_summaries(probes, labels=labels)
    outputs = write_comparison(args.output_dir, comparison)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

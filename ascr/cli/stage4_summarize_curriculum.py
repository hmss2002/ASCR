import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(summary, key):
    if key in summary:
        return summary.get(key)
    return (summary.get("metrics") or {}).get(key)


def summarize_curriculum(summary_paths, labels=None):
    summary_paths = [Path(path) for path in summary_paths]
    labels = list(labels or [])
    if labels and len(labels) != len(summary_paths):
        raise ValueError("--labels must have same length as --summaries")
    rows = []
    for index, path in enumerate(summary_paths):
        summary = _read_json(path)
        label = labels[index] if labels else f"grid{summary.get('grid_size', index)}"
        rows.append({
            "label": label,
            "summary_path": str(path),
            "grid_size": summary.get("grid_size"),
            "input_mode": summary.get("input_mode"),
            "target_schema": summary.get("target_schema"),
            "row_count": summary.get("row_count"),
            "parse_rate": summary.get("parse_rate"),
            "parsed_count": summary.get("parsed_count"),
            "malformed_count": summary.get("malformed_count"),
            "call_error_count": summary.get("call_error_count"),
            "hit_any_rate": _metric(summary, "hit_any_rate"),
            "mean_f1_at_k": _metric(summary, "mean_f1_at_k"),
            "mean_iou": _metric(summary, "mean_iou"),
            "mean_distance_to_target_cells": _metric(summary, "mean_distance_to_target_cells"),
            "mean_latency_ms": summary.get("mean_latency_ms"),
        })
    best_hit = max((row["hit_any_rate"] or 0.0) for row in rows) if rows else 0.0
    best_parse = max((row["parse_rate"] or 0.0) for row in rows) if rows else 0.0
    return {
        "schema_version": "ascr.stage4.curriculum_summary.v1",
        "created_at_utc": _created_at(),
        "rows": rows,
        "best_hit_any_rate": best_hit,
        "best_parse_rate": best_parse,
    }


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_outputs(output_dir, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "curriculum_summary.json"
    markdown_path = output_dir / "curriculum_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-4 Curriculum Summary",
        "",
        "| Label | Grid | Parse | Hit any | F1 | IoU | Distance | Rows | Malformed | Latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {label} | {grid} | {parse} | {hit} | {f1} | {iou} | {dist} | {rows} | {malformed} | {latency} |".format(
                label=row["label"],
                grid=_fmt(row["grid_size"]),
                parse=_fmt(row["parse_rate"]),
                hit=_fmt(row["hit_any_rate"]),
                f1=_fmt(row["mean_f1_at_k"]),
                iou=_fmt(row["mean_iou"]),
                dist=_fmt(row["mean_distance_to_target_cells"]),
                rows=_fmt(row["row_count"]),
                malformed=_fmt(row["malformed_count"]),
                latency=_fmt(row["mean_latency_ms"]),
            )
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"summary_json": str(summary_path), "summary_md": str(markdown_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize Stage-4 coarse-to-fine MMU LoRA curriculum probe summaries.")
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = summarize_curriculum(args.summaries, labels=args.labels)
    outputs = write_outputs(args.output_dir, summary)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

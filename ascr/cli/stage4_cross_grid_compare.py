"""Compare Stage-4 probe summaries across grid resolutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["_source_path"] = str(path)
    return payload


def compare_summaries(paths, labels=None):
    labels = labels or [Path(path).parent.name for path in paths]
    rows = []
    for path, label in zip(paths, labels):
        payload = _read_json(path)
        metrics = payload.get("metrics") or {}
        rows.append({
            "label": label,
            "summary": str(path),
            "grid_size": payload.get("grid_size"),
            "row_count": payload.get("row_count"),
            "parse_rate": payload.get("parse_rate"),
            "hit_any_rate": metrics.get("hit_any_rate"),
            "mean_f1_at_k": metrics.get("mean_f1_at_k"),
            "mean_iou": metrics.get("mean_iou"),
            "malformed_count": payload.get("malformed_count"),
            "call_error_count": payload.get("call_error_count"),
        })
    best = sorted(rows, key=lambda row: (float(row.get("hit_any_rate") or 0.0), float(row.get("parse_rate") or 0.0)), reverse=True)
    return {
        "schema_version": "ascr.stage4.cross_grid_compare.v1",
        "row_count": len(rows),
        "rows": rows,
        "best_label": best[0]["label"] if best else None,
    }


def write_outputs(output_dir, comparison):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cross_grid_comparison.json"
    md_path = output_dir / "cross_grid_comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-4 Cross-Grid Comparison",
        "",
        "| Label | Grid | Rows | Parse | Hit any | F1 | IoU | Malformed | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| {row['label']} | {row.get('grid_size')} | {row.get('row_count')} | {row.get('parse_rate')} | "
            f"{row.get('hit_any_rate')} | {row.get('mean_f1_at_k')} | {row.get('mean_iou')} | "
            f"{row.get('malformed_count')} | {row.get('call_error_count')} |"
        )
    lines.extend(["", f"Best label: `{comparison.get('best_label')}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"comparison_json": str(json_path), "comparison_md": str(md_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Compare Stage-4 probe summaries across grids.")
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    comparison = compare_summaries(args.summaries, labels=args.labels)
    print(json.dumps(write_outputs(args.output_dir, comparison), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


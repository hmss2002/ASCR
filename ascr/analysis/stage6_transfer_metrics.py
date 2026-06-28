"""Metrics for synthetic-to-real Stage-6 transfer checks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path):
    rows = []
    if not path or not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _labels(row, *keys):
    for key in keys:
        value = row.get(key)
        if value:
            if isinstance(value, str):
                return {value}
            return {str(item) for item in value}
    return set()


def _rate(rows, predicate):
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def summarize_transfer_gap(synthetic_rows, transfer_rows):
    synthetic_hit = _rate(
        synthetic_rows,
        lambda row: bool(_labels(row, "predicted_cells", "lora_cells") & _labels(row, "target_cells", "coarse_labels")),
    )
    synthetic_nonempty = _rate(synthetic_rows, lambda row: bool(_labels(row, "predicted_cells", "lora_cells")))
    transfer_nonempty = _rate(transfer_rows, lambda row: bool(_labels(row, "predicted_cells", "lora_cells", "cells")))
    transfer_parse = _rate(transfer_rows, lambda row: row.get("status") in {"parsed", "ok", None} and bool(_labels(row, "predicted_cells", "lora_cells", "cells")))
    return {
        "schema_version": "ascr.stage6.transfer_metrics.v1",
        "created_at_utc": created_at_utc(),
        "synthetic_row_count": len(synthetic_rows),
        "transfer_row_count": len(transfer_rows),
        "synthetic_hit_any_rate": synthetic_hit,
        "synthetic_nonempty_rate": synthetic_nonempty,
        "transfer_nonempty_rate": transfer_nonempty,
        "transfer_parse_nonempty_rate": transfer_parse,
        "transfer_gap_nonempty": synthetic_nonempty - transfer_nonempty,
        "transfer_gap_hit_proxy": synthetic_hit - transfer_nonempty,
    }


def write_outputs(output_dir, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage6_transfer_metrics.json"
    md_path = output_dir / "stage6_transfer_metrics.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-6 Transfer Metrics",
        "",
        f"- Synthetic rows: `{summary['synthetic_row_count']}`",
        f"- Transfer rows: `{summary['transfer_row_count']}`",
        f"- Synthetic hit-any rate: `{summary['synthetic_hit_any_rate']:.4f}`",
        f"- Synthetic nonempty rate: `{summary['synthetic_nonempty_rate']:.4f}`",
        f"- Transfer nonempty rate: `{summary['transfer_nonempty_rate']:.4f}`",
        f"- Transfer gap proxy: `{summary['transfer_gap_hit_proxy']:.4f}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"metrics_json": str(json_path), "metrics_md": str(md_path)}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Compute Stage-6 synthetic-to-real transfer metrics.")
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--transfer-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    summary = summarize_transfer_gap(
        read_jsonl(args.synthetic_manifest),
        read_jsonl(args.transfer_manifest),
    )
    print(json.dumps(write_outputs(args.output_dir, summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

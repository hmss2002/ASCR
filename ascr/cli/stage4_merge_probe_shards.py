"""Merge Stage-4 MMU probe shards into one summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ascr.training.stage3_selectors import evaluate_predictions, write_json, write_jsonl


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_probe_shards(shard_dirs, output_dir, label="merged_stage4_probe"):
    shard_dirs = [Path(path) for path in shard_dirs]
    summaries = []
    probe_rows = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        rows_path = shard_dir / "probe_rows.jsonl"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = _read_json(summary_path)
        summary["summary_path"] = str(summary_path)
        summaries.append(summary)
        for row in _read_jsonl(rows_path):
            row["shard_dir"] = str(shard_dir)
            probe_rows.append(row)
    if not probe_rows:
        raise ValueError(f"No probe_rows.jsonl found in shard dirs: {[str(path) for path in shard_dirs]}")
    first = summaries[0] if summaries else probe_rows[0]
    grid_size = int(first.get("grid_size") or probe_rows[0].get("grid_size") or 16)
    top_k = int(first.get("top_k") or 4)
    prediction_map = {str(row.get("sample_id")): row.get("predicted_cells") or [] for row in probe_rows}
    examples = [
        {
            "sample_id": str(row.get("sample_id")),
            "prompt": row.get("prompt", ""),
            "corruption_type": row.get("corruption_type", "unknown"),
            "target_cells": row.get("target_cells") or [],
        }
        for row in probe_rows
    ]
    metrics, predictions = evaluate_predictions(
        examples,
        prediction_map,
        grid_size=grid_size,
        baseline=label,
        top_k=top_k,
    )
    row_count = len(probe_rows)
    parsed_count = sum(1 for row in probe_rows if row.get("status") == "parsed")
    malformed_count = sum(1 for row in probe_rows if row.get("status") == "abstained_malformed_json")
    call_error_count = sum(1 for row in probe_rows if row.get("status") == "call_error")
    latencies = [float(row.get("latency_ms") or 0.0) for row in probe_rows]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "ascr.stage4.mmu_localization_probe.merged_summary.v1",
        "created_at_utc": created_at_utc(),
        "label": label,
        "output_dir": str(output_dir),
        "shard_count": len(summaries),
        "shards": summaries,
        "row_count": row_count,
        "parsed_count": parsed_count,
        "malformed_count": malformed_count,
        "call_error_count": call_error_count,
        "parse_rate": parsed_count / row_count if row_count else 0.0,
        "grid_size": grid_size,
        "top_k": top_k,
        "input_mode": first.get("input_mode"),
        "target_schema": first.get("target_schema"),
        "prompt_variant": first.get("prompt_variant"),
        "lora_path": first.get("lora_path"),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "metrics": metrics,
    }
    write_jsonl(output_dir / "probe_rows.jsonl", probe_rows)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Merge Stage-4 MMU localization probe shard outputs.")
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="merged_stage4_probe")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = merge_probe_shards(args.shard_dirs, args.output_dir, label=args.label)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

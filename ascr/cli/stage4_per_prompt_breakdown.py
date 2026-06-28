"""Break down Stage-4 localization quality by prompt type."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_tags(prompt):
    text = str(prompt or "").lower()
    tags = []
    word_count = len(text.split())
    tags.append("short" if word_count <= 8 else "long" if word_count >= 18 else "medium")
    if any(token in text for token in ("left", "right", "above", "below", "behind", "front", "between", "near")):
        tags.append("spatial")
    if any(token in text for token in ("red", "blue", "green", "yellow", "black", "white", "purple", "orange")):
        tags.append("color")
    if any(token in text for token in ("text", "letter", "sign", "word", "logo")):
        tags.append("text")
    if " and " in text or " with " in text or "," in text:
        tags.append("compositional")
    if len(tags) == 1:
        tags.append("simple")
    return tags


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for tag in prompt_tags(row.get("prompt")):
            buckets[tag].append(row)
    out = []
    for tag, tag_rows in sorted(buckets.items()):
        parsed = [row for row in tag_rows if row.get("status") == "parsed"]
        hit_any = []
        for row in tag_rows:
            target = {str(item) for item in row.get("target_cells") or []}
            pred = {str(item) for item in row.get("predicted_cells") or []}
            hit_any.append(bool(target & pred))
        out.append({
            "tag": tag,
            "row_count": len(tag_rows),
            "parse_rate": len(parsed) / len(tag_rows) if tag_rows else 0.0,
            "hit_any_rate": sum(1 for value in hit_any if value) / len(tag_rows) if tag_rows else 0.0,
            "mean_predicted_cells": sum(len(row.get("predicted_cells") or []) for row in tag_rows) / len(tag_rows) if tag_rows else 0.0,
        })
    return {"schema_version": "ascr.stage4.per_prompt_breakdown.v1", "rows": out}


def write_outputs(output_dir, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "per_prompt_breakdown.json"
    md_path = output_dir / "per_prompt_breakdown.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-4 Per-Prompt Breakdown",
        "",
        "| Tag | Rows | Parse | Hit any | Mean predicted cells |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(f"| {row['tag']} | {row['row_count']} | {row['parse_rate']:.4f} | {row['hit_any_rate']:.4f} | {row['mean_predicted_cells']:.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"breakdown_json": str(json_path), "breakdown_md": str(md_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize Stage-4 probe rows by prompt type.")
    parser.add_argument("--probe-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(json.dumps(write_outputs(args.output_dir, summarize(read_jsonl(args.probe_rows))), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


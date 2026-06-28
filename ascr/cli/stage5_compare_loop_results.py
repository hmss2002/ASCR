"""Summarize Stage-5 loop before/after manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_loop_manifest(rows):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    mask_counts = [int((row.get("mask_stats") or {}).get("selected_token_count") or 0) for row in ok_rows]
    cell_counts = [len(row.get("lora_cells") or []) for row in ok_rows]
    changed = [bool(row.get("reopen_changed")) for row in ok_rows]
    return {
        "schema_version": "ascr.stage5.loop_comparison.v1",
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "mask_nonempty_rate": sum(1 for count in mask_counts if count > 0) / len(ok_rows) if ok_rows else 0.0,
        "reopen_change_rate": sum(1 for value in changed if value) / len(ok_rows) if ok_rows else 0.0,
        "mean_cells_selected_per_sample": sum(cell_counts) / len(ok_rows) if ok_rows else 0.0,
        "mean_tokens_selected_per_sample": sum(mask_counts) / len(ok_rows) if ok_rows else 0.0,
    }


def write_outputs(output_dir, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage5_loop_comparison.json"
    md_path = output_dir / "stage5_loop_comparison.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# Stage-5 Loop Comparison",
            "",
            f"- Rows: `{summary['row_count']}`",
            f"- OK rows: `{summary['ok_count']}`",
            f"- Mask nonempty rate: `{summary['mask_nonempty_rate']:.4f}`",
            f"- Reopen change rate: `{summary['reopen_change_rate']:.4f}`",
            f"- Mean cells selected: `{summary['mean_cells_selected_per_sample']:.4f}`",
            f"- Mean tokens selected: `{summary['mean_tokens_selected_per_sample']:.4f}`",
            "",
        ]),
        encoding="utf-8",
    )
    return {"comparison_json": str(json_path), "comparison_md": str(md_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Compare Stage-5 self-corruption loop outputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = summarize_loop_manifest(read_jsonl(args.manifest))
    print(json.dumps(write_outputs(args.output_dir, summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


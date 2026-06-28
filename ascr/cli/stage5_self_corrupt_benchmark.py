"""Batch runner for Stage-5 self-corruption ASCR loops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ascr.cli.stage5_self_corrupt_loop import run_stage5_loop, write_json
from ascr.core.config import load_config


def read_prompts(path, limit=None):
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            prompts.append(text)
        if limit is not None and len(prompts) >= int(limit):
            break
    return prompts


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    return str(path)


def run_benchmark(args):
    config = load_config(args.config) if args.config else {}
    prompts = read_prompts(args.prompts, limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for index, prompt in enumerate(prompts):
        sample_dir = output_dir / f"sample_{index:04d}"
        try:
            trace = run_stage5_loop(
                prompt,
                sample_dir,
                config=config,
                seed=int(args.seed) + index,
                corruption_type=args.corruption_type or config.get("corruption_type", "block_4x4_random_replace"),
                grid_size=int(args.grid_size or config.get("grid_size", 4)),
                max_selected_cells=int(args.max_selected_cells or config.get("max_selected_cells", 4)),
                lora_path=args.lora_path or config.get("lora_path"),
                mock=bool(args.mock or config.get("mock", False)),
            )
            row = {
                "sample_index": index,
                "domain": args.domain,
                "prompt": prompt,
                "trace": str(sample_dir / "trace.json"),
                "clean_image": trace["clean_image"],
                "corrupted_image": trace["corrupted_image"],
                "repaired_image": trace["repaired_image"],
                "target_cells": trace["target_cells"],
                "lora_cells": trace["lora_cells"],
                "mask_stats": trace["mask_stats"],
                "reopen_changed": trace["reopen_changed"],
                "status": "ok",
            }
            rows.append(row)
        except Exception as exc:
            error = {"sample_index": index, "prompt": prompt, "status": "error", "error_type": exc.__class__.__name__, "error": str(exc)}
            errors.append(error)
            rows.append(error)
            if not args.keep_going:
                raise
    manifest = write_jsonl(output_dir / "manifest.jsonl", rows)
    summary = {
        "schema_version": "ascr.stage5.self_corrupt_benchmark.v1",
        "domain": args.domain,
        "prompt_count": len(prompts),
        "row_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "error_count": len(errors),
        "manifest": manifest,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Batch Stage-5 self-corruption ASCR loops.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--domain", default="hard64_self_corrupt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=1, help="Reserved for future multi-iteration loops.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corruption-type", default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--max-selected-cells", type=int, default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--mock", action="store_true")
    return parser


def main(argv=None):
    summary = run_benchmark(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


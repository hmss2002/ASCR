#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

DEFAULT_REPO = "sayakpaul/drawbench"
DEFAULT_FILENAME = "DrawBench Prompts - Sheet1.csv"
TARGET_CATEGORIES = ["Colors", "Counting", "Positional", "Conflicting", "Text", "Descriptions", "DALL-E", "Rare Words"]

def download_drawbench(raw_dir: Path, repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download DrawBench prompts") from exc
    raw_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id, filename=filename, repo_type="dataset", local_dir=str(raw_dir)))

def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for index, row in enumerate(rows):
        prompt = (row.get("Prompts") or row.get("prompt") or "").strip()
        category = (row.get("Category") or row.get("category") or "unknown").strip()
        if prompt:
            normalized.append({"source": "drawbench", "index": index, "category": category, "prompt": prompt})
    return normalized

def build_smoke(rows, categories, limit):
    selected = []
    seen = set()
    for row in rows:
        category = row["category"]
        if category in categories and category not in seen:
            selected.append(row)
            seen.add(category)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for row in rows:
            if row not in selected:
                selected.append(row)
            if len(selected) >= limit:
                break
    return selected[:limit]

def write_prompt_file(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row["prompt"] + "\n" for row in rows), encoding="utf-8")

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Download DrawBench prompt metadata and export ASCR prompt-suite text files.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--raw-dir", default="data/benchmarks/raw/drawbench")
    parser.add_argument("--processed-jsonl", default="data/benchmarks/processed/drawbench_prompts.jsonl")
    parser.add_argument("--all-output", default="configs/benchmarks/prompts/drawbench_all.txt")
    parser.add_argument("--smoke-output", default="configs/benchmarks/prompts/drawbench_smoke8.txt")
    parser.add_argument("--smoke-limit", type=int, default=8)
    args = parser.parse_args(argv)
    candidate = Path(args.raw_dir) / args.filename
    csv_path = candidate if candidate.exists() else download_drawbench(Path(args.raw_dir), args.repo_id, args.filename)
    rows = load_rows(csv_path)
    smoke_rows = build_smoke(rows, TARGET_CATEGORIES, args.smoke_limit)
    write_jsonl(Path(args.processed_jsonl), rows)
    write_prompt_file(Path(args.all_output), rows)
    write_prompt_file(Path(args.smoke_output), smoke_rows)
    print(json.dumps({
        "csv_path": str(csv_path),
        "prompt_count": len(rows),
        "smoke_count": len(smoke_rows),
        "all_output": args.all_output,
        "smoke_output": args.smoke_output,
        "smoke_categories": [row["category"] for row in smoke_rows],
    }, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

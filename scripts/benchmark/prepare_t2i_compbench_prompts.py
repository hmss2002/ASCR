#!/usr/bin/env python
import argparse
import json
from pathlib import Path

DEFAULT_REPO = "NinaKarine/t2i-compbench"
CATEGORY_FILES = {
    "color": "color_val/val-00000-of-00001.parquet",
    "shape": "shape_val/val-00000-of-00001.parquet",
    "texture": "texture_val/val-00000-of-00001.parquet",
    "spatial": "spatial_val/val-00000-of-00001.parquet",
    "3d_spatial": "3d_spatial_val/spatial_val-00000-of-00001.parquet",
    "numeracy": "numeracy_val/val-00000-of-00001.parquet",
    "complex": "complex_val/val-00000-of-00001.parquet",
    "complex_spatial": "complex_val_spatial/val_spatial-00000-of-00001.parquet",
    "complex_action": "complex_val_action/val_action-00000-of-00001.parquet",
    "complex_spatialaction": "complex_val_spatialaction/val_spatialaction-00000-of-00001.parquet",
}
HARD_CATEGORY_ORDER = [
    "color",
    "shape",
    "texture",
    "spatial",
    "3d_spatial",
    "numeracy",
    "complex",
    "complex_spatial",
]


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required. Install it with: python -m pip install pyarrow") from exc
    return pq


def download_parquet(repo_id, filename, raw_dir):
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download T2I-CompBench prompts") from exc
    raw_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id, filename=filename, repo_type="dataset", local_dir=str(raw_dir)))


def read_rows(parquet_path, category, source_file, pq):
    table = pq.read_table(parquet_path, columns=["text"])
    rows = []
    seen = set()
    for index, text in enumerate(table.column("text").to_pylist()):
        prompt = str(text or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        rows.append({
            "source": "t2i-compbench",
            "repo": DEFAULT_REPO,
            "category": category,
            "source_file": source_file,
            "index": index,
            "prompt": prompt,
        })
    return rows


def round_robin_select(rows, categories, limit):
    by_category = {category: [] for category in categories}
    for row in rows:
        category = row["category"]
        if category in by_category:
            by_category[category].append(row)
    selected = []
    seen_prompts = set()
    cursors = {category: 0 for category in categories}
    while len(selected) < limit:
        made_progress = False
        for category in categories:
            bucket = by_category.get(category, [])
            cursor = cursors[category]
            while cursor < len(bucket) and bucket[cursor]["prompt"] in seen_prompts:
                cursor += 1
            cursors[category] = cursor
            if cursor < len(bucket):
                row = bucket[cursor]
                selected.append(row)
                seen_prompts.add(row["prompt"])
                cursors[category] = cursor + 1
                made_progress = True
                if len(selected) >= limit:
                    break
        if not made_progress:
            break
    return selected


def write_prompt_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(row["prompt"] for row in rows) + chr(10), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download T2I-CompBench prompt metadata and export ASCR prompt-suite files.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--raw-dir", default="data/benchmarks/raw/t2i_compbench")
    parser.add_argument("--processed-jsonl", default="data/benchmarks/processed/t2i_compbench_prompts.jsonl")
    parser.add_argument("--hard-output", default="configs/benchmarks/prompts/t2i_compbench_hard64.txt")
    parser.add_argument("--smoke-output", default="configs/benchmarks/prompts/t2i_compbench_hard_smoke8.txt")
    parser.add_argument("--hard-limit", type=int, default=64)
    parser.add_argument("--smoke-limit", type=int, default=8)
    args = parser.parse_args(argv)

    pq = require_pyarrow()
    raw_dir = Path(args.raw_dir)
    rows = []
    for category, filename in CATEGORY_FILES.items():
        parquet_path = download_parquet(args.repo_id, filename, raw_dir)
        rows.extend(read_rows(parquet_path, category, filename, pq))

    hard_rows = round_robin_select(rows, HARD_CATEGORY_ORDER, args.hard_limit)
    smoke_rows = round_robin_select(rows, HARD_CATEGORY_ORDER, args.smoke_limit)
    write_jsonl(Path(args.processed_jsonl), rows)
    write_prompt_file(Path(args.hard_output), hard_rows)
    write_prompt_file(Path(args.smoke_output), smoke_rows)
    print(json.dumps({
        "repo_id": args.repo_id,
        "prompt_count": len(rows),
        "hard_count": len(hard_rows),
        "smoke_count": len(smoke_rows),
        "hard_output": args.hard_output,
        "smoke_output": args.smoke_output,
        "categories": HARD_CATEGORY_ORDER,
        "smoke_prompts": smoke_rows,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Prepare benchmark data for DPG-Bench, GenAI-Bench, and DSG-1k.

Outputs (relative to ASCR project root):
  configs/benchmark_data/dpg_bench.csv         - copy of DPG-Bench CSV (local)
  configs/benchmark_data/dsg1k_anns.csv        - DSG-1k questions CSV (from GitHub)
  configs/benchmark_data/genai_bench.jsonl     - GenAI-Bench prompts + skills (from HF)
  configs/prompts/dpg_bench_1065.txt           - DPG-Bench unique prompts (one per line)
  configs/prompts/dsg1k_1060.txt               - DSG-1k unique prompts (one per line)
  configs/prompts/genai_bench_1600.txt         - GenAI-Bench prompts (one per line)
  configs/prompts/bench3_combined.txt          - all 3725 prompts combined
  configs/benchmark_data/bench3_index.json     - maps {item_id -> {benchmark, prompt, line_no}}
"""

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DPG_SRC = PROJECT_ROOT / "external/Show-o/show-o2/evaluation/dpg_bench.csv"
DSG_URL = "https://raw.githubusercontent.com/j-min/DSG/main/dsg/data/dsg-1k-anns.csv"
GENAI_HF_DATASET = "BaiqiL/GenAI-Bench"

OUT_DATA = PROJECT_ROOT / "configs/benchmark_data"
OUT_PROMPTS = PROJECT_ROOT / "configs/prompts"


def extract_dpg(src_csv: Path, out_csv: Path, out_prompts: Path) -> dict:
    """Extract DPG-Bench prompts, copy CSV, return {dpg_{item_id}: prompt}."""
    rows = list(csv.DictReader(src_csv.open(encoding="utf-8")))
    # Copy CSV
    out_csv.write_text(src_csv.read_text(encoding="utf-8"), encoding="utf-8")

    # Collect unique prompts in order of first occurrence; prefix to avoid collisions
    seen = {}
    for row in rows:
        item_id = f"dpg_{row['item_id']}"
        if item_id not in seen:
            seen[item_id] = row["text"]

    prompts_text = "\n".join(seen.values()) + "\n"
    out_prompts.write_text(prompts_text, encoding="utf-8")
    print(f"[DPG-Bench] {len(seen)} unique prompts, {len(rows)} questions → {out_prompts.name}")
    return seen


def download_dsg(out_csv: Path, out_prompts: Path) -> dict:
    """Download DSG-1k CSV, extract prompts, return {dsg_{item_id}: prompt}."""
    print(f"[DSG-1k] Downloading from GitHub...")
    with urllib.request.urlopen(DSG_URL, timeout=30) as resp:
        content = resp.read().decode("utf-8")
    out_csv.write_text(content, encoding="utf-8")

    rows = list(csv.DictReader(content.splitlines()))
    seen = {}
    for row in rows:
        item_id = f"dsg_{row['item_id']}"
        if item_id not in seen:
            seen[item_id] = row["text"]

    prompts_text = "\n".join(seen.values()) + "\n"
    out_prompts.write_text(prompts_text, encoding="utf-8")
    print(f"[DSG-1k] {len(seen)} unique prompts, {len(rows)} questions → {out_prompts.name}")

    # Per-category breakdown
    cats = {}
    for item_id in seen:
        cat = item_id.split("_")[1]  # dsg_whoops -> whoops
        cats[cat] = cats.get(cat, 0) + 1
    print(f"[DSG-1k] Categories: {cats}")
    return seen


def download_genai(out_jsonl: Path, out_prompts: Path) -> dict:
    """Download GenAI-Bench from HuggingFace, return {genai_{item_id}: prompt}."""
    print(f"[GenAI-Bench] Loading from HuggingFace ({GENAI_HF_DATASET})...")
    try:
        from datasets import load_dataset
        ds = load_dataset(GENAI_HF_DATASET, split="test")
    except Exception:
        try:
            from datasets import load_dataset
            ds = load_dataset(GENAI_HF_DATASET)
            # pick first split
            split_name = list(ds.keys())[0]
            ds = ds[split_name]
        except Exception as e:
            print(f"[GenAI-Bench] datasets library failed: {e}")
            print("[GenAI-Bench] Trying huggingface_hub parquet download...")
            return _download_genai_via_hub(out_jsonl, out_prompts)

    records = {}
    lines = []
    for item in ds:
        item_id = f"genai_{item.get('Index', len(records))}"
        prompt = item["Prompt"]
        tags = item.get("Tags", {})
        records[item_id] = prompt
        lines.append(json.dumps({
            "item_id": item_id,
            "prompt": prompt,
            "basic_skills": tags.get("basic_skills", []) if isinstance(tags, dict) else [],
            "advanced_skills": tags.get("advanced_skills", []) if isinstance(tags, dict) else [],
        }))

    out_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_prompts.write_text("\n".join(records.values()) + "\n", encoding="utf-8")
    print(f"[GenAI-Bench] {len(records)} prompts → {out_prompts.name}")
    return records


def _download_genai_via_hub(out_jsonl: Path, out_prompts: Path) -> dict:
    """Fallback: download GenAI-Bench parquet via huggingface_hub."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    parquet_path = hf_hub_download(
        repo_id=GENAI_HF_DATASET,
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path, columns=["Index", "Prompt", "Tags"])
    records = {}
    lines = []
    for _, row in df.iterrows():
        item_id = f"genai_{row['Index']}"
        prompt = row["Prompt"]
        tags = row.get("Tags", {}) or {}
        records[item_id] = prompt
        lines.append(json.dumps({
            "item_id": item_id,
            "prompt": prompt,
            "basic_skills": tags.get("basic_skills", []) if isinstance(tags, dict) else [],
            "advanced_skills": tags.get("advanced_skills", []) if isinstance(tags, dict) else [],
        }))
    out_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_prompts.write_text("\n".join(records.values()) + "\n", encoding="utf-8")
    print(f"[GenAI-Bench] {len(records)} prompts → {out_prompts.name}")
    return records


def build_combined(dpg: dict, dsg: dict, genai: dict, out_prompts: Path, out_index: Path):
    """Build combined prompts file and index mapping."""
    # Combined order: DPG, DSG, GenAI
    index = {}
    lines = []
    line_no = 0

    for item_id, prompt in dpg.items():
        index[item_id] = {"benchmark": "dpg", "prompt": prompt, "line_no": line_no}
        lines.append(prompt)
        line_no += 1

    for item_id, prompt in dsg.items():
        index[item_id] = {"benchmark": "dsg", "prompt": prompt, "line_no": line_no}
        lines.append(prompt)
        line_no += 1

    for item_id, prompt in genai.items():
        index[item_id] = {"benchmark": "genai", "prompt": prompt, "line_no": line_no}
        lines.append(prompt)
        line_no += 1

    out_prompts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_index.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[Combined] {line_no} total prompts → {out_prompts.name}")
    print(f"[Combined] Index → {out_index.name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data for DPG/GenAI/DSG.")
    parser.add_argument("--skip-genai", action="store_true", help="Skip GenAI-Bench (if HF unavailable)")
    parser.add_argument("--skip-dsg", action="store_true", help="Skip DSG-1k (if GitHub unavailable)")
    args = parser.parse_args()

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_PROMPTS.mkdir(parents=True, exist_ok=True)

    # DPG-Bench
    dpg = extract_dpg(
        DPG_SRC,
        OUT_DATA / "dpg_bench.csv",
        OUT_PROMPTS / "dpg_bench_1065.txt",
    )

    # DSG-1k
    if args.skip_dsg:
        dsg = {}
        print("[DSG-1k] Skipped.")
    else:
        dsg = download_dsg(
            OUT_DATA / "dsg1k_anns.csv",
            OUT_PROMPTS / "dsg1k_1060.txt",
        )

    # GenAI-Bench
    if args.skip_genai:
        genai = {}
        print("[GenAI-Bench] Skipped.")
    else:
        genai = download_genai(
            OUT_DATA / "genai_bench.jsonl",
            OUT_PROMPTS / "genai_bench_1600.txt",
        )

    # Combined
    build_combined(
        dpg, dsg, genai,
        OUT_PROMPTS / "bench3_combined.txt",
        OUT_DATA / "bench3_index.json",
    )

    print("\n✅ Data preparation complete.")
    print(f"  DPG-Bench:   {len(dpg)} prompts")
    print(f"  DSG-1k:      {len(dsg)} prompts")
    print(f"  GenAI-Bench: {len(genai)} prompts")
    print(f"  Combined:    {len(dpg)+len(dsg)+len(genai)} prompts")


if __name__ == "__main__":
    main()

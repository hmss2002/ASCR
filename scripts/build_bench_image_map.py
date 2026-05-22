#!/usr/bin/env python3
"""
Build a unified image map for all 3 benchmarks.

Reads:
  - ShowO+ASCR merged suite.json (from geneval_gen_shard pipeline)
  - BAGEL shard suite.json files (from bench_bagel_gen_shard pipeline)
  - bench3_index.json (maps item_id -> {benchmark, prompt, line_no})

Outputs:
  - image_map.json: {item_id: {showo: path, ascr: path, bagel: path, prompt: str, benchmark: str}}

Usage:
  python scripts/build_bench_image_map.py \
    --showo-suite outputs/bench3_showo_YYYYMMDD/suite.json \
    --bagel-run-root outputs/bench3_bagel_YYYYMMDD \
    --bench3-index configs/benchmark_data/bench3_index.json \
    --output outputs/bench3_image_map_YYYYMMDD.json
"""

import argparse
import json
from pathlib import Path


def load_showo_suite(suite_path: Path) -> dict:
    """Load ShowO+ASCR suite.json and return {prompt: {showo, ascr}}."""
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    results = data.get("results", [data]) if "results" in data else [data]
    by_prompt = {}
    for r in results:
        prompt = r.get("prompt", "").strip()
        if prompt:
            by_prompt[prompt] = {
                "showo": r.get("baseline_image", ""),
                "ascr": r.get("ascr_final_image", ""),
            }
    return by_prompt


def load_bagel_shards(bagel_run_root: Path) -> dict:
    """Load all BAGEL shard suite.json files and return {prompt: path}."""
    shard_dirs = sorted(bagel_run_root.glob("shard_*/suite.json"))
    by_prompt = {}
    loaded = 0
    for suite_path in shard_dirs:
        data = json.loads(suite_path.read_text(encoding="utf-8"))
        results = data.get("results", []) if isinstance(data, dict) and "results" in data else [data]
        for r in results:
            prompt = r.get("prompt", "").strip()
            if prompt and "bagel_image" in r:
                by_prompt[prompt] = r["bagel_image"]
                loaded += 1
    print(f"[BAGEL] Loaded {loaded} entries from {len(shard_dirs)} shards")
    return by_prompt


def main():
    parser = argparse.ArgumentParser(description="Build bench3 image map from generation outputs.")
    parser.add_argument("--showo-suite", required=True, help="Path to merged ShowO+ASCR suite.json")
    parser.add_argument("--bagel-run-root", required=True, help="BAGEL shards run root (contains shard_*/)")
    parser.add_argument("--bench3-index", default="configs/benchmark_data/bench3_index.json",
                        help="bench3_index.json from prepare_bench_data.py")
    parser.add_argument("--output", required=True, help="Output image_map.json path")
    args = parser.parse_args()

    index = json.loads(Path(args.bench3_index).read_text(encoding="utf-8"))

    showo_map = load_showo_suite(Path(args.showo_suite))
    bagel_map = load_bagel_shards(Path(args.bagel_run_root))

    print(f"[ShowO/ASCR] Loaded {len(showo_map)} entries")
    print(f"[BAGEL]      Loaded {len(bagel_map)} entries")
    print(f"[Index]      {len(index)} expected entries")

    image_map = {}
    missing_showo = []
    missing_bagel = []

    for item_id, meta in index.items():
        prompt = meta["prompt"]
        showo_entry = showo_map.get(prompt, {})
        bagel_path = bagel_map.get(prompt, "")

        if not showo_entry:
            missing_showo.append(item_id)
        if not bagel_path:
            missing_bagel.append(item_id)

        image_map[item_id] = {
            "prompt": prompt,
            "benchmark": meta["benchmark"],
            "showo": showo_entry.get("showo", ""),
            "ascr": showo_entry.get("ascr", ""),
            "bagel": bagel_path,
        }

    print(f"\n[Summary]")
    print(f"  Total items:      {len(image_map)}")
    print(f"  Missing ShowO:    {len(missing_showo)}")
    print(f"  Missing BAGEL:    {len(missing_bagel)}")

    # Benchmark breakdown
    benchmarks = {}
    for v in image_map.values():
        b = v["benchmark"]
        benchmarks[b] = benchmarks.get(b, 0) + 1
    print(f"  By benchmark: {benchmarks}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(image_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n✅ Image map written to: {output_path}")


if __name__ == "__main__":
    main()

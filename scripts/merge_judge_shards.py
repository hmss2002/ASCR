#!/usr/bin/env python3
"""Merge sharded outputs from judge_showo_ascr_pairwise_qwen.py into one report.

Usage:
  python scripts/merge_judge_shards.py SHARD0.json SHARD1.json ... --output FINAL.json
"""
import argparse, json
from collections import Counter
from datetime import datetime
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    shards = []
    for p in args.shards:
        shards.append(json.loads(Path(p).read_text(encoding="utf-8")))

    # Sanity-check labels & shard cardinality
    num_shards = shards[0].get("num_shards", len(shards))
    labels = {s.get("baseline_label") for s in shards}, {s.get("ascr_label") for s in shards}
    if len(labels[0]) != 1 or len(labels[1]) != 1:
        raise SystemExit(f"Inconsistent labels across shards: {labels}")

    all_records = []
    counts = Counter()
    seen_prompts = set()
    for s in shards:
        for r in s.get("records", []):
            all_records.append(r)
            counts[r.get("pairwise_verdict", "unknown")] += 1
            seen_prompts.add(r.get("prompt"))

    all_records.sort(key=lambda r: r.get("prompt", ""))

    report = {
        "protocol": shards[0].get("protocol", "qwen_clean_side_by_side_pairwise_judge_v1") + "_merged",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "merged_from": [str(p) for p in args.shards],
        "num_shards": num_shards,
        "config": shards[0].get("config"),
        "baseline_label": shards[0].get("baseline_label"),
        "ascr_label": shards[0].get("ascr_label"),
        "prompt_count": len(all_records),
        "counts": dict(counts),
        "records": all_records,
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    # Write a tiny .md summary
    md = outp.with_suffix(".md")
    lines = [
        f"# Pairwise judge (merged): {report['baseline_label']} (LEFT) vs {report['ascr_label']} (RIGHT)",
        f"- prompts: {report['prompt_count']}",
        f"- counts: {report['counts']}",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(outp), "markdown": str(md), "counts": report["counts"], "prompt_count": report["prompt_count"]}, indent=2))

if __name__ == "__main__":
    main()

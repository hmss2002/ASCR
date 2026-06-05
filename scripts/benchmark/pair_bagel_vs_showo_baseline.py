#!/usr/bin/env python3
"""
Pair BAGEL images (already generated) with ShowO baseline images from an ASCR run.
Creates a suite.json compatible with judge_showo_ascr_pairwise_qwen.py.

Usage:
    python scripts/benchmark/pair_bagel_vs_showo_baseline.py \
        --bagel-suite outputs/bagel_.../bagel_vs_ascr_suite.json \
        --ascr-run-root outputs/benchmarks_.../ \
        --output outputs/bagel_.../bagel_vs_showo_baseline_suite.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


def dir_to_prompt(dir_name: str) -> str:
    """Convert 'prompt_NNN-a-green-bench-...' to 'a green bench ...'"""
    parts = dir_name.split("-", 1)
    if len(parts) == 2 and parts[0].startswith("prompt_"):
        return parts[1].replace("-", " ")
    return dir_name.replace("-", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagel-suite", required=True,
                        help="Path to bagel_vs_ascr_suite.json")
    parser.add_argument("--ascr-run-root", required=True,
                        help="Root dir of ASCR run containing shard_*/ subdirs")
    parser.add_argument("--output", required=True,
                        help="Output suite JSON path")
    args = parser.parse_args()

    # Load BAGEL suite
    with open(args.bagel_suite, encoding="utf-8") as f:
        bagel_suite = json.load(f)
    bagel_results = bagel_suite.get("results", bagel_suite)
    if not isinstance(bagel_results, list):
        bagel_results = list(bagel_results.values())

    # Build BAGEL image map: prompt -> image path
    bagel_map = {}
    for r in bagel_results:
        prompt = r["prompt"]
        image = (r.get("baseline_image") or r.get("image") or r.get("bagel_image"))
        if image:
            bagel_map[prompt] = image
    print(f"Loaded {len(bagel_map)} BAGEL images")

    # Find ShowO baseline images from ASCR run root
    ascr_root = Path(args.ascr_run_root)
    baseline_images = sorted(ascr_root.rglob("baseline_showo.png"))
    print(f"Found {len(baseline_images)} baseline_showo.png files")

    # Build map: prompt -> ShowO baseline image path
    showo_map: dict[str, str] = {}
    for img_path in baseline_images:
        prompt_text = dir_to_prompt(img_path.parent.name)
        showo_map[prompt_text] = str(img_path)
    print(f"Mapped {len(showo_map)} ShowO baseline images by prompt")

    # Match by prompt
    records = []
    missing = []
    for prompt, bagel_image in sorted(bagel_map.items()):
        showo_image = showo_map.get(prompt)
        if showo_image is None:
            missing.append(prompt)
            print(f"  WARNING: no ShowO baseline for: {prompt!r}")
            continue
        records.append({
            "prompt": prompt,
            "baseline_image": showo_image,    # ShowO = LEFT
            "ascr_final_image": bagel_image,  # BAGEL = RIGHT
        })

    print(f"Paired {len(records)} records; {len(missing)} unmatched")
    if missing:
        print("Unmatched:", missing)

    suite = {
        "protocol": "bagel_vs_showo_baseline_pairwise_v1",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "bagel_suite": args.bagel_suite,
        "ascr_run_root": args.ascr_run_root,
        "prompt_count": len(records),
        "results": records,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

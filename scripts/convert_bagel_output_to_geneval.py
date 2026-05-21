#!/usr/bin/env python3
"""
Convert BAGEL generation suite.json to GenEval directory format.

GenEval directory structure:
    {outdir}/{idx}/
        metadata.jsonl   (single JSON line)
        samples/
            0.png

Usage:
    python scripts/convert_bagel_output_to_geneval.py \
        --suite-json outputs/geneval_bagel_.../suite.json \
        --metadata configs/geneval_metadata.jsonl \
        --outdir outputs/geneval_bagel_.../geneval_bagel
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-json", required=True, help="Path to BAGEL suite.json")
    parser.add_argument("--metadata", default="configs/geneval_metadata.jsonl",
                        help="Path to geneval metadata JSONL")
    parser.add_argument("--outdir", required=True, help="Output GenEval directory")
    parser.add_argument("--seed", type=int, default=0, help="Sample index (default: 0)")
    args = parser.parse_args()

    suite = json.loads(Path(args.suite_json).read_text(encoding="utf-8"))
    results = suite.get("results", [])
    metadata = [
        json.loads(line)
        for line in Path(args.metadata).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    prompt_to_idx = {m["prompt"]: i for i, m in enumerate(metadata)}
    outdir = Path(args.outdir)

    print(f"Loaded {len(results)} BAGEL results, {len(metadata)} geneval metadata entries")

    matched = skipped = 0
    for rec in results:
        prompt = rec.get("prompt", "")
        idx = prompt_to_idx.get(prompt)
        if idx is None:
            print(f"  WARNING: prompt not in geneval metadata: {prompt!r}")
            skipped += 1
            continue

        img_path = Path(rec.get("image") or rec.get("bagel_image") or "")
        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}")
            skipped += 1
            continue

        dst = outdir / str(idx) / "samples" / f"{args.seed}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if img_path.suffix.lower() == ".png":
            shutil.copy2(img_path, dst)
        else:
            from PIL import Image
            Image.open(img_path).convert("RGB").save(dst, "PNG")

        meta_dst = outdir / str(idx) / "metadata.jsonl"
        meta_dst.write_text(json.dumps(metadata[idx]) + "\n", encoding="utf-8")
        matched += 1

    print(f"Done: {matched} matched, {skipped} skipped.")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()

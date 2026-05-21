#!/usr/bin/env python3
"""
Convert ASCR run output (aggregated suite.json) to GenEval directory format.

GenEval directory structure:
    {outdir}/{idx}/
        metadata.jsonl     (single JSON, despite .jsonl extension)
        samples/
            0.png          (seed 0)

Usage:
    python scripts/convert_showo_output_to_geneval.py \
        --suite-json outputs/geneval_run_.../suite.json \
        --metadata configs/geneval_metadata.jsonl \
        --baseline-outdir outputs/geneval_run_.../geneval_baseline \
        --ascr-outdir outputs/geneval_run_.../geneval_ascr
"""
import argparse
import json
import shutil
from pathlib import Path


def load_suite(suite_path: Path):
    """Load suite.json and return list of result records."""
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        results = data.get("results", [])
    else:
        results = data
    return results


def load_metadata(metadata_path: Path):
    """Load geneval metadata JSONL and return list of metadata dicts."""
    return [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image(path_str: str) -> Path | None:
    """Resolve image path, handling relative paths and PPM -> PNG conversion."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    # Try relative to cwd
    p2 = Path.cwd() / path_str
    if p2.exists():
        return p2
    return None


def copy_image_as_png(src: Path, dst: Path) -> bool:
    """Copy image to dst as PNG. Converts PPM/etc. if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return True
    # Convert via PIL
    try:
        from PIL import Image
        with Image.open(src) as img:
            img.convert("RGB").save(dst, "PNG")
        return True
    except Exception as exc:
        print(f"  WARNING: failed to convert {src} -> {dst}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-json", required=True, help="Path to aggregated suite.json")
    parser.add_argument("--metadata", default="configs/geneval_metadata.jsonl",
                        help="Path to geneval metadata JSONL")
    parser.add_argument("--baseline-outdir", required=True,
                        help="Output directory for ShowO baseline images in geneval format")
    parser.add_argument("--ascr-outdir", required=True,
                        help="Output directory for ASCR images in geneval format")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed index to use as sample name (default: 0)")
    args = parser.parse_args()

    suite_path = Path(args.suite_json)
    metadata_path = Path(args.metadata)
    baseline_dir = Path(args.baseline_outdir)
    ascr_dir = Path(args.ascr_outdir)

    results = load_suite(suite_path)
    metadata = load_metadata(metadata_path)

    # Build prompt -> (idx, meta) map from geneval metadata
    prompt_to_idx = {m["prompt"]: i for i, m in enumerate(metadata)}
    print(f"Loaded {len(metadata)} geneval metadata entries")
    print(f"Loaded {len(results)} suite results")

    matched = 0
    skipped = 0
    for result in results:
        prompt = result.get("prompt", "")
        idx = prompt_to_idx.get(prompt)
        if idx is None:
            print(f"  WARNING: prompt not found in geneval metadata: {prompt!r}")
            skipped += 1
            continue

        meta = metadata[idx]
        idx_str = str(idx)

        # Baseline image
        baseline_src = find_image(result.get("baseline_image", ""))
        if baseline_src:
            dst = baseline_dir / idx_str / "samples" / f"{args.seed}.png"
            if copy_image_as_png(baseline_src, dst):
                meta_dst = baseline_dir / idx_str / "metadata.jsonl"
                meta_dst.write_text(json.dumps(meta) + "\n", encoding="utf-8")

        # ASCR image
        ascr_src_path = (
            result.get("ascr_final_image") or
            result.get("accepted_ascr_image") or
            result.get("final_decoded_image")
        )
        ascr_src = find_image(ascr_src_path or "")
        if ascr_src:
            dst = ascr_dir / idx_str / "samples" / f"{args.seed}.png"
            if copy_image_as_png(ascr_src, dst):
                meta_dst = ascr_dir / idx_str / "metadata.jsonl"
                meta_dst.write_text(json.dumps(meta) + "\n", encoding="utf-8")

        if baseline_src and ascr_src:
            matched += 1
        else:
            missing = []
            if not baseline_src:
                missing.append("baseline")
            if not ascr_src:
                missing.append("ascr")
            print(f"  MISSING {missing} for idx={idx} prompt={prompt!r}")
            skipped += 1

    print(f"\nConversion complete: {matched} matched, {skipped} skipped/warned")
    print(f"  Baseline geneval dir: {baseline_dir}")
    print(f"  ASCR geneval dir: {ascr_dir}")


if __name__ == "__main__":
    main()

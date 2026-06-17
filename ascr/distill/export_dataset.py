import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_raw(record):
    return {key: value for key, value in record.items() if key != "raw_text"}


def export_dataset(distill_dir, output, include_raw_text=False):
    root = Path(distill_dir)
    output = Path(output)
    quality_by_idx = {}
    for record in read_jsonl(root / "quality_labels.jsonl"):
        quality_by_idx[int(record["idx"])] = record if include_raw_text else strip_raw(record)
    localization_by_idx = defaultdict(list)
    for record in read_jsonl(root / "localization_labels.jsonl"):
        localization_by_idx[int(record["idx"])].append(record if include_raw_text else strip_raw(record))
    rows = []
    for idx in sorted(set(quality_by_idx) | set(localization_by_idx)):
        quality = quality_by_idx.get(idx)
        localizations = sorted(localization_by_idx.get(idx, []), key=lambda item: int(item.get("iteration", 0)))
        prompt = (quality or localizations[0]).get("prompt") if (quality or localizations) else ""
        rows.append({
            "schema_version": "ascr.teacher_dataset.v1",
            "idx": idx,
            "sample_id": f"p{idx:03d}",
            "prompt": prompt,
            "quality": quality,
            "localizations": localizations,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    manifest = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "distill_dir": str(root),
        "output": str(output),
        "include_raw_text": bool(include_raw_text),
        "row_count": len(rows),
        "quality_count": len(quality_by_idx),
        "localization_count": sum(len(values) for values in localization_by_idx.values()),
    }
    manifest_path = output.with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Export ASCR teacher distillation labels as a clean training JSONL dataset.")
    parser.add_argument("--distill-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-raw-text", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = export_dataset(args.distill_dir, args.output, include_raw_text=args.include_raw_text)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

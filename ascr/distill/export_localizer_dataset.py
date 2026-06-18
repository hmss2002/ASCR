import argparse
from datetime import datetime
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def strip_raw(payload):
    if isinstance(payload, dict):
        return {key: strip_raw(value) for key, value in payload.items() if key != "raw_text"}
    if isinstance(payload, list):
        return [strip_raw(value) for value in payload]
    return payload


def base_rows(base_dataset):
    rows = []
    for row in read_jsonl(base_dataset):
        clean = strip_raw(row)
        clean.setdefault("schema_version", "ascr.localizer_dataset.v1")
        clean.setdefault("domain", "hard64")
        clean.setdefault("source", "base_teacher_dataset")
        for localization in clean.get("localizations", []) or []:
            localization.setdefault("domain", clean.get("domain"))
            localization.setdefault("source", clean.get("source"))
        rows.append(clean)
    return rows


def extra_rows(paths):
    rows = []
    for path in paths:
        for record in read_jsonl(path):
            clean = strip_raw(record)
            prompt = clean.get("prompt", "")
            sample_id = clean.get("sample_id")
            rows.append({
                "schema_version": "ascr.localizer_dataset.v1",
                "sample_id": sample_id,
                "idx": None,
                "prompt": prompt,
                "domain": clean.get("domain") or "unknown",
                "source": "image_manifest_teacher_localization",
                "quality": None,
                "localizations": [{
                    "schema_version": clean.get("schema_version"),
                    "sample_id": sample_id,
                    "source_sample_id": clean.get("source_sample_id"),
                    "prompt": prompt,
                    "domain": clean.get("domain") or "unknown",
                    "source": "image_manifest_teacher_localization",
                    "image_field": clean.get("image_field"),
                    "grid_image": clean.get("grid_image"),
                    "teacher_model": clean.get("teacher_model"),
                    "student_model": clean.get("student_model"),
                    "evaluation": clean.get("evaluation", {}),
                }],
            })
    return rows


def export_dataset(base_dataset, extra_localizations, output):
    rows = []
    if base_dataset:
        rows.extend(base_rows(base_dataset))
    rows.extend(extra_rows(extra_localizations or []))
    output = Path(output)
    write_jsonl(output, rows)
    manifest = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "schema_version": "ascr.localizer_dataset_manifest.v1",
        "base_dataset": str(base_dataset) if base_dataset else None,
        "extra_localizations": [str(path) for path in extra_localizations or []],
        "output": str(output),
        "row_count": len(rows),
        "localization_count": sum(len(row.get("localizations", []) or []) for row in rows),
        "domains": sorted({str(row.get("domain", "unknown")) for row in rows}),
    }
    manifest_path = output.with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Merge ASCR localizer teacher labels into one training dataset.")
    parser.add_argument("--base-dataset", default=None)
    parser.add_argument("--extra-localizations", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = export_dataset(args.base_dataset, args.extra_localizations, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

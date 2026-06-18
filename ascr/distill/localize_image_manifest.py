import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from ascr.core.schemas import safe_parse_semantic_evaluation
from ascr.distill.api_client import DEFAULT_MODEL, api_settings, build_client, chat_completion_text
from ascr.distill.teacher import (
    LOCALIZATION_SCHEMA_TEXT,
    extract_json_object_with_repair,
    localization_messages,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id:
            done.add(sample_id)
    return done


def resolve_path(path, manifest_path):
    raw = Path(path)
    if raw.is_absolute():
        return raw
    candidates = [
        Path.cwd() / raw,
        Path(manifest_path).resolve().parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_image_fields(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def iter_tasks(manifest, image_fields, limit=None):
    rows = read_jsonl(manifest)
    if limit is not None:
        rows = rows[: int(limit)]
    for row in rows:
        for field in image_fields:
            image_path = row.get(field)
            if not image_path:
                continue
            yield {
                "sample_id": f"{row.get('sample_id')}:{field}",
                "source_sample_id": row.get("sample_id"),
                "domain": row.get("domain"),
                "prompt": row.get("prompt", ""),
                "image_field": field,
                "grid_image": image_path,
                "student_model": row.get("student_model"),
                "manifest_row": row,
            }


def run_task(task, manifest_path, client, model, grid_size, max_selected_cells, max_tokens, retries, repair_retries, include_raw_text=False):
    resolved_image = resolve_path(task["grid_image"], manifest_path)
    raw_text = chat_completion_text(
        client,
        localization_messages(
            task["prompt"],
            resolved_image,
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
            compact=True,
        ),
        model=model,
        max_tokens=max_tokens,
        retries=retries,
    )
    payload = extract_json_object_with_repair(
        raw_text,
        client,
        model,
        LOCALIZATION_SCHEMA_TEXT,
        repair_retries=repair_retries,
    )
    evaluation = safe_parse_semantic_evaluation(payload, grid_size=grid_size, max_selected_cells=max_selected_cells)
    record = {
        "schema_version": "ascr.image_manifest_localization.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "kind": "localization",
        "sample_id": task["sample_id"],
        "source_sample_id": task["source_sample_id"],
        "domain": task.get("domain"),
        "prompt": task["prompt"],
        "image_field": task["image_field"],
        "grid_image": task["grid_image"],
        "student_model": task.get("student_model"),
        "teacher_model": model,
        "evaluation": evaluation.to_dict(),
    }
    if include_raw_text:
        record["raw_text"] = raw_text
    return record


def run_localize_manifest(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "localization_labels.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if args.overwrite:
        for path in (labels_path, errors_path):
            if path.exists():
                path.unlink()
    settings = api_settings()
    model = args.model or settings["model"] or DEFAULT_MODEL
    max_tokens = int(args.max_tokens or os.environ.get("ASCR_TEACHER_LOCALIZATION_MAX_TOKENS", 2048))
    repair_retries = int(args.repair_retries or os.environ.get("ASCR_TEACHER_JSON_REPAIR_RETRIES", 1))
    client = build_client(base_url=args.base_url or settings["base_url"])
    done = completed_ids(labels_path)
    tasks = list(iter_tasks(args.manifest, parse_image_fields(args.image_fields), limit=args.limit))
    attempted = 0
    for task in tasks:
        if task["sample_id"] in done and not args.overwrite:
            continue
        attempted += 1
        try:
            record = run_task(
                task,
                args.manifest,
                client,
                model,
                grid_size=args.grid_size,
                max_selected_cells=args.max_selected_cells,
                max_tokens=max_tokens,
                retries=args.retries,
                repair_retries=repair_retries,
                include_raw_text=args.include_raw_text,
            )
            append_jsonl(labels_path, record)
        except Exception as exc:
            append_jsonl(errors_path, {
                "sample_id": task["sample_id"],
                "source_sample_id": task["source_sample_id"],
                "domain": task.get("domain"),
                "prompt": task["prompt"],
                "image_field": task["image_field"],
                "grid_image": task["grid_image"],
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:1000],
            })
            if not args.keep_going:
                raise
    labels = read_jsonl(labels_path) if labels_path.exists() else []
    errors = read_jsonl(errors_path) if errors_path.exists() else []
    manifest = {
        "schema_version": "ascr.image_manifest_localization.manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "model": model,
        "task_count": len(tasks),
        "attempted_count": attempted,
        "label_count": len(labels),
        "error_count": len(errors),
        "image_fields": parse_image_fields(args.image_fields),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Use Qwen3.7/OFOX to localize semantic errors for image benchmark manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-fields", default="before_grid_image,after_grid_image")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--repair-retries", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-selected-cells", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-raw-text", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_localize_manifest(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

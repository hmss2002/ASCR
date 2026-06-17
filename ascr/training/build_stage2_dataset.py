import argparse
import json
from pathlib import Path

from ascr.core.schemas import GridCell


DATASET_SCHEMA_VERSION = "stage2.teacher_trace.v1"


def build_parser():
    parser = argparse.ArgumentParser(description="Build a Stage 2 teacher-trace dataset from ASCR trace.jsonl outputs.")
    parser.add_argument("input_roots", nargs="+", help="One or more output roots to scan for trace.jsonl files.")
    parser.add_argument("--output", default=None, help="Destination JSONL file. Optional for --dry-run.")
    parser.add_argument("--skipped-report", default=None, help="Optional JSONL file for malformed trace records.")
    parser.add_argument("--trace-glob", default="**/trace.jsonl", help="Glob used to locate trace files under each input root.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and summarize without writing dataset files.")
    return parser


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_trace_files(input_roots, trace_glob):
    seen = set()
    for root in input_roots:
        for trace_path in sorted(Path(root).glob(trace_glob)):
            resolved = trace_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield trace_path


def _normalize_cells(regions, grid_size):
    labels = []
    for region in regions or []:
        for raw_cell in region.get("cells", []) or []:
            cell = GridCell.from_any(raw_cell, grid_size)
            labels.append(cell.to_label())
    deduped = []
    seen = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped


def _sample_from_record(record, next_record, trace_path, summary):
    evaluation = record["evaluation"]
    mask = record["reopen_mask"]
    artifact_paths = record["artifact_paths"]
    raw = evaluation.get("raw") or {}
    teacher_payload = raw.get("ofox_payload") or raw.get("qwen_vl_payload") or evaluation
    teacher_raw = raw.get("ofox_raw_text") or raw.get("qwen_vl_text") or raw.get("raw_text") or ""
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "prompt": record["original_prompt"],
        "iteration": int(record["iteration"]),
        "decoded_image": artifact_paths.get("decoded_image"),
        "grid_image": artifact_paths.get("grid_image"),
        "teacher_raw": teacher_raw,
        "teacher_json": teacher_payload,
        "selected_4x4_cells": _normalize_cells(evaluation.get("regions", []), int(mask.get("coarse_grid_size", 4) or 4)),
        "projected_token_mask": mask,
        "token_grid_size": int(mask.get("token_grid_size", 64)),
        "coarse_grid_size": 4,
        "dilation": 1,
        "correction_instruction": evaluation.get("correction_instruction", ""),
        "before_image": artifact_paths.get("decoded_image"),
        "after_image": (next_record or {}).get("artifact_paths", {}).get("decoded_image"),
        "stop_reason": summary.get("stop_reason"),
        "revision_gain": ((record.get("reserved_for_stage2") or {}).get("revision_gain")),
        "trace_path": str(trace_path),
    }


def build_dataset(input_roots, trace_glob="**/trace.jsonl"):
    dataset = []
    skipped = []
    for trace_path in _iter_trace_files(input_roots, trace_glob):
        lines = trace_path.read_text(encoding="utf-8").splitlines()
        summary = _read_json(trace_path.parent / "summary.json")
        parsed_records = []
        for line_number, line in enumerate(lines, start=1):
            try:
                parsed_records.append(json.loads(line))
            except Exception as exc:
                skipped.append({"trace_path": str(trace_path), "line_number": line_number, "error": f"invalid_json: {exc}"})
        for index, record in enumerate(parsed_records):
            try:
                sample = _sample_from_record(
                    record=record,
                    next_record=parsed_records[index + 1] if index + 1 < len(parsed_records) else None,
                    trace_path=trace_path,
                    summary=summary,
                )
                dataset.append(sample)
            except Exception as exc:
                skipped.append({
                    "trace_path": str(trace_path),
                    "line_number": index + 1,
                    "iteration": record.get("iteration"),
                    "error": str(exc),
                })
    return dataset, skipped


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
    return path


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset, skipped = build_dataset(args.input_roots, trace_glob=args.trace_glob)
    payload = {
        "status": "ok",
        "input_roots": [str(Path(root)) for root in args.input_roots],
        "trace_glob": args.trace_glob,
        "dataset_records": len(dataset),
        "skipped_records": len(skipped),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry-run is used")
    if not args.dry_run:
        payload["output"] = str(_write_jsonl(args.output, dataset))
        skipped_report = args.skipped_report or (str(Path(args.output).with_suffix("")) + "_skipped.jsonl")
        payload["skipped_report"] = str(_write_jsonl(skipped_report, skipped))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

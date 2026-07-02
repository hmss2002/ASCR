"""Validate and summarize Stage-3 clean-token manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_jsonl(path):
    rows = []
    errors = []
    path = Path(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append({"path": str(path), "line": line_number, "error": str(exc)})
    return rows, errors


def _normalise_path(path, project_root=None):
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if project_root:
        return (Path(project_root) / raw).resolve()
    return raw


def _count_values(rows, key):
    counts = Counter()
    missing = 0
    for row in rows:
        value = row.get(key)
        if value is None:
            missing += 1
        else:
            counts[str(value)] += 1
    return {"missing": missing, "counts": dict(sorted(counts.items()))}


def _duplicates(rows, key, limit):
    counts = Counter(str(row.get(key)) for row in rows if row.get(key) is not None)
    duplicates = [(value, count) for value, count in counts.items() if count > 1]
    duplicates.sort(key=lambda item: (-item[1], item[0]))
    return {
        "duplicate_count": len(duplicates),
        "examples": [{"value": value, "count": count} for value, count in duplicates[:limit]],
    }


def _token_files(output_root):
    if not output_root:
        return []
    root = Path(output_root)
    if not root.exists():
        return []
    return sorted(path.resolve() for path in root.glob("**/clean_tokens/clean_p*_vq_ids.json") if path.is_file())


def _summary_files(output_root):
    if not output_root:
        return []
    root = Path(output_root)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("**/summary.json") if path.is_file())


def _read_summaries(output_root, limit):
    summaries = []
    errors = []
    for path in _summary_files(output_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            summaries.append(payload)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    elapsed_values = [float(item["elapsed_s"]) for item in summaries if item.get("elapsed_s") is not None]
    row_count = sum(int(item.get("row_count") or 0) for item in summaries)
    generated_count = sum(int(item.get("generated_count") or 0) for item in summaries)
    skipped_existing_count = sum(int(item.get("skipped_existing_count") or 0) for item in summaries)
    elapsed_total = sum(elapsed_values) if elapsed_values else None
    rows_with_timing = sum(int(item.get("row_count") or 0) for item in summaries if item.get("elapsed_s") is not None)
    generated_with_timing = sum(int(item.get("generated_count") or 0) for item in summaries if item.get("elapsed_s") is not None)
    return {
        "summary_file_count": len(summaries),
        "summary_error_count": len(errors),
        "summary_errors": errors[:limit],
        "summary_row_count": row_count,
        "summary_generated_count": generated_count,
        "summary_skipped_existing_count": skipped_existing_count,
        "summary_elapsed_s_total": elapsed_total,
        "summary_rows_per_s": (rows_with_timing / elapsed_total) if elapsed_total else None,
        "summary_generated_per_s": (generated_with_timing / elapsed_total) if elapsed_total else None,
        "summary_files_with_timing": len(elapsed_values),
        "summary_examples": summaries[:limit],
    }


def build_clean_manifest_report(
    manifests,
    *,
    output_root=None,
    project_root=None,
    min_rows=0,
    example_limit=20,
):
    rows = []
    json_errors = []
    manifest_paths = [Path(path) for path in manifests]
    for manifest in manifest_paths:
        manifest_rows, manifest_errors = _read_jsonl(manifest)
        rows.extend(manifest_rows)
        json_errors.extend(manifest_errors)

    referenced_paths = []
    missing_paths = []
    for row in rows:
        clean_path = row.get("clean_vq_ids_path")
        if not clean_path:
            missing_paths.append({"sample_id": row.get("sample_id"), "path": None, "reason": "missing clean_vq_ids_path"})
            continue
        normalized = _normalise_path(clean_path, project_root=project_root)
        referenced_paths.append(normalized.resolve())
        if not normalized.exists():
            missing_paths.append({"sample_id": row.get("sample_id"), "path": str(normalized), "reason": "file not found"})

    referenced_set = {path for path in referenced_paths}
    discovered_files = _token_files(output_root)
    unmanifested = [path for path in discovered_files if path not in referenced_set]
    summary_report = _read_summaries(output_root, example_limit)

    prompt_indexes = [row.get("prompt_index") for row in rows if row.get("prompt_index") is not None]
    duplicate_samples = _duplicates(rows, "sample_id", example_limit)
    duplicate_prompts = _duplicates(rows, "prompt_index", example_limit)
    failures = []
    if json_errors:
        failures.append("json_errors")
    if len(rows) < int(min_rows):
        failures.append("row_count_below_min")
    if missing_paths:
        failures.append("missing_clean_vq_files")
    if duplicate_samples["duplicate_count"]:
        failures.append("duplicate_sample_ids")
    if duplicate_prompts["duplicate_count"]:
        failures.append("duplicate_prompt_indexes")

    return {
        "schema_version": "ascr.stage3.clean_manifest_report.v1",
        "created_at_utc": _created_at(),
        "manifests": [str(path) for path in manifest_paths],
        "output_root": str(output_root) if output_root else None,
        "project_root": str(project_root) if project_root else None,
        "min_rows": int(min_rows),
        "ok": not failures,
        "failures": failures,
        "row_count": len(rows),
        "manifest_count": len(manifest_paths),
        "json_error_count": len(json_errors),
        "json_errors": json_errors[:example_limit],
        "unique_sample_id_count": len({row.get("sample_id") for row in rows if row.get("sample_id") is not None}),
        "duplicate_sample_ids": duplicate_samples,
        "unique_prompt_index_count": len(set(prompt_indexes)),
        "prompt_index_min": min(prompt_indexes, default=None),
        "prompt_index_max": max(prompt_indexes, default=None),
        "duplicate_prompt_indexes": duplicate_prompts,
        "referenced_clean_vq_file_count": len(referenced_paths),
        "missing_clean_vq_file_count": len(missing_paths),
        "missing_clean_vq_files": missing_paths[:example_limit],
        "discovered_clean_vq_file_count": len(discovered_files),
        "unmanifested_clean_vq_file_count": len(unmanifested),
        "unmanifested_clean_vq_files": [str(path) for path in unmanifested[:example_limit]],
        **summary_report,
        "token_grid_size": _count_values(rows, "token_grid_size"),
        "image_size": _count_values(rows, "image_size"),
        "generation_timesteps": _count_values(rows, "generation_timesteps"),
        "guidance_scale": _count_values(rows, "guidance_scale"),
        "temperature": _count_values(rows, "temperature"),
    }


def write_outputs(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage3_clean_manifest_report.json"
    md_path = output_dir / "stage3_clean_manifest_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-3 Clean Manifest Report",
        "",
        f"Status: `{'ok' if report.get('ok') else 'failed'}`",
        f"Rows: `{report.get('row_count')}`",
        f"Manifest files: `{report.get('manifest_count')}`",
        f"Referenced token files: `{report.get('referenced_clean_vq_file_count')}`",
        f"Discovered token files: `{report.get('discovered_clean_vq_file_count')}`",
        f"Missing token files: `{report.get('missing_clean_vq_file_count')}`",
        f"Unmanifested token files: `{report.get('unmanifested_clean_vq_file_count')}`",
        f"Duplicate sample IDs: `{report.get('duplicate_sample_ids', {}).get('duplicate_count')}`",
        f"Duplicate prompt indexes: `{report.get('duplicate_prompt_indexes', {}).get('duplicate_count')}`",
        f"Summary files: `{report.get('summary_file_count')}`",
        f"Summary rows/s: `{report.get('summary_rows_per_s') or ''}`",
        "",
        "Failures:",
    ]
    failures = report.get("failures") or []
    if failures:
        lines.extend(f"- `{failure}`" for failure in failures)
    else:
        lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"clean_manifest_report_json": str(json_path), "clean_manifest_report_md": str(md_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Validate and summarize Stage-3 clean-token manifests.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when critical validation fails.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = build_clean_manifest_report(
        args.manifests,
        output_root=args.output_root,
        project_root=args.project_root,
        min_rows=args.min_rows,
    )
    if args.output_dir:
        outputs = write_outputs(args.output_dir, report)
        print(json.dumps(outputs, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

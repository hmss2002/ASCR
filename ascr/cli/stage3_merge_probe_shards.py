import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import re

from ascr.analysis.stage3_self_corrupt import write_json, write_jsonl


PATH_FIELDS = (
    "clean_vq_ids_path",
    "corrupted_vq_ids_path",
    "clean_image",
    "corrupted_image",
)


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _candidate_exists(value, project_root=None):
    if not value:
        return False
    path = Path(value)
    candidates = [path]
    if project_root and not path.is_absolute():
        candidates.append(Path(project_root) / path)
    return any(candidate.exists() for candidate in candidates)


def _parse_sample_id(sample_id):
    match = re.fullmatch(r"p(\d+)_c(\d+)", str(sample_id or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _normalise_row_ids(row, summary):
    prompt_offset = int(summary.get("prompt_offset") or 0)
    parsed = _parse_sample_id(row.get("sample_id"))
    if row.get("prompt_index") is not None:
        prompt_index = int(row["prompt_index"])
    elif parsed:
        prompt_index = prompt_offset + parsed[0]
    else:
        prompt_index = None
    corruption_index = parsed[1] if parsed else int(row.get("corruption_index") or 0)
    if prompt_index is not None:
        row["prompt_index"] = prompt_index
        row["sample_id"] = f"p{prompt_index:04d}_c{corruption_index:03d}"
    return row


def _sort_key(row):
    parsed = _parse_sample_id(row.get("sample_id"))
    if parsed:
        return parsed
    return (int(row.get("prompt_index") or 0), int(row.get("corruption_index") or 0))


def merge_probe_shards(shard_dirs, output_dir, project_root=".", allow_missing_paths=False):
    output_dir = Path(output_dir)
    project_root = Path(project_root) if project_root else None
    shard_infos = []
    shard_summaries = []
    rows = []
    seen_ids = set()
    duplicate_ids = []
    missing_paths = []
    for shard_dir in sorted(Path(path) for path in shard_dirs):
        summary_path = shard_dir / "summary.json"
        manifest_path = shard_dir / "manifest.jsonl"
        if not summary_path.exists() or not manifest_path.exists():
            shard_infos.append({
                "shard_dir": str(shard_dir),
                "status": "missing_manifest_or_summary",
                "summary_exists": summary_path.exists(),
                "manifest_exists": manifest_path.exists(),
            })
            continue
        summary = _read_json(summary_path)
        shard_summaries.append(summary)
        shard_rows = _read_jsonl(manifest_path)
        for row in shard_rows:
            row = _normalise_row_ids(dict(row), summary)
            row["source_shard_dir"] = str(shard_dir)
            row["source_shard_summary"] = str(summary_path)
            sample_id = str(row.get("sample_id"))
            if sample_id in seen_ids:
                duplicate_ids.append(sample_id)
            seen_ids.add(sample_id)
            for field in PATH_FIELDS:
                if row.get(field) and not _candidate_exists(row[field], project_root=project_root):
                    missing_paths.append({"sample_id": sample_id, "field": field, "path": row[field]})
            rows.append(row)
        shard_infos.append({
            "shard_dir": str(shard_dir),
            "status": "ok",
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "prompt_offset": summary.get("prompt_offset"),
            "prompt_limit": summary.get("prompt_limit"),
            "prompt_count": summary.get("prompt_count"),
            "row_count": len(shard_rows),
        })
    if duplicate_ids:
        raise ValueError(f"Duplicate sample_id values across shards: {sorted(set(duplicate_ids))[:10]}")
    if missing_paths and not allow_missing_paths:
        raise ValueError(f"Missing referenced paths in merged shards: {missing_paths[:5]}")
    rows.sort(key=_sort_key)
    prompt_indices = sorted({int(row["prompt_index"]) for row in rows if row.get("prompt_index") is not None})
    corruption_types = sorted({
        ((row.get("corruption") or {}).get("corruption_type") or row.get("corruption_type") or "unknown")
        for row in rows
    })
    analysis_grids = sorted({
        int(metric["grid_size"])
        for row in rows
        for metric in (row.get("metrics") or [])
        if metric.get("grid_size") is not None
    })
    token_grid_sizes = sorted({
        int((row.get("corruption") or {}).get("token_grid_size"))
        for row in rows
        if (row.get("corruption") or {}).get("token_grid_size") is not None
    })
    image_sizes = sorted({
        int(summary["image_size"])
        for summary in shard_summaries
        if summary.get("image_size") is not None
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(manifest_path, rows)
    summary = {
        "schema_version": "ascr.stage3.token_locality_probe.merged_summary.v1",
        "created_at_utc": _created_at(),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "shard_count": len(shard_infos),
        "ok_shard_count": sum(1 for shard in shard_infos if shard["status"] == "ok"),
        "row_count": len(rows),
        "prompt_count": len(prompt_indices),
        "prompt_indices": prompt_indices,
        "prompt_index_min": min(prompt_indices) if prompt_indices else None,
        "prompt_index_max": max(prompt_indices) if prompt_indices else None,
        "corruption_types": corruption_types,
        "analysis_grids": analysis_grids,
        "token_grid_size": token_grid_sizes[0] if len(token_grid_sizes) == 1 else None,
        "image_size": image_sizes[0] if len(image_sizes) == 1 else None,
        "missing_referenced_paths": missing_paths,
        "shards": shard_infos,
    }
    write_json(summary_path, summary)
    return summary


def _expand_shard_args(values):
    expanded = []
    for value in values:
        matches = sorted(glob.glob(value)) if any(ch in value for ch in "*?[") else []
        expanded.extend(Path(match) for match in matches)
        if not matches:
            expanded.append(Path(value))
    return expanded


def build_parser():
    parser = argparse.ArgumentParser(description="Merge Stage-3 token locality probe shard outputs.")
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--allow-missing-paths", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = merge_probe_shards(
        _expand_shard_args(args.shard_dirs),
        args.output_dir,
        project_root=args.project_root,
        allow_missing_paths=args.allow_missing_paths,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

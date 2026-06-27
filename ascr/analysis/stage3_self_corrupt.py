"""Model-light Stage-3 self-corruption dataset and report helpers."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median


REPORT_SCHEMA_VERSION = "ascr.stage3.locality_report.v1"
DATASET_ROW_SCHEMA_VERSION = "ascr.stage3.self_corrupt_dataset.row.v1"
DATASET_MANIFEST_SCHEMA_VERSION = "ascr.stage3.self_corrupt_dataset_manifest.v1"


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    return str(path)


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _hit_rate(values):
    values = [bool(value) for value in values if value is not None]
    return sum(1 for value in values if value) / len(values) if values else None


def _corruption_type(row):
    corruption = row.get("corruption") or {}
    return corruption.get("corruption_type") or row.get("corruption_type") or "unknown"


def aggregate_locality(rows):
    grouped = defaultdict(list)
    corruption_totals = defaultdict(list)
    for row in rows:
        corruption_type = _corruption_type(row)
        for metric in row.get("metrics", []) or []:
            key = (corruption_type, int(metric["grid_size"]))
            grouped[key].append(metric)
            corruption_totals[corruption_type].append(metric)

    by_group = []
    for (corruption_type, grid_size), metrics in sorted(grouped.items()):
        radii = [metric.get("effective_radius_cells") for metric in metrics if metric.get("effective_radius_cells") is not None]
        by_group.append({
            "corruption_type": corruption_type,
            "grid_size": grid_size,
            "row_count": len(metrics),
            "mean_inside_energy_fraction": _mean(metric.get("inside_energy_fraction") for metric in metrics),
            "mean_inside_outside_energy_ratio": _mean(metric.get("inside_outside_energy_ratio") for metric in metrics),
            "mean_center_displacement_cells": _mean(metric.get("center_displacement_cells") for metric in metrics),
            "top1_hit_rate": _hit_rate(metric.get("top1_cell_hit") for metric in metrics),
            "topk_hit_rate": _hit_rate(metric.get("topk_cell_hit") for metric in metrics),
            "median_effective_radius_cells": float(median(radii)) if radii else None,
        })

    by_corruption = []
    for corruption_type, metrics in sorted(corruption_totals.items()):
        by_corruption.append({
            "corruption_type": corruption_type,
            "metric_count": len(metrics),
            "mean_inside_energy_fraction": _mean(metric.get("inside_energy_fraction") for metric in metrics),
            "top1_hit_rate": _hit_rate(metric.get("top1_cell_hit") for metric in metrics),
            "topk_hit_rate": _hit_rate(metric.get("topk_cell_hit") for metric in metrics),
        })
    return by_group, by_corruption


def build_locality_report(manifest_path, summary_path=None):
    manifest_path = Path(manifest_path)
    rows = read_jsonl(manifest_path)
    summary = read_json(summary_path) if summary_path else {}
    by_group, by_corruption = aggregate_locality(rows)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": created_at_utc(),
        "manifest": str(manifest_path),
        "summary": str(summary_path) if summary_path else None,
        "row_count": len(rows),
        "prompt_count": summary.get("prompt_count"),
        "analysis_grids": summary.get("analysis_grids"),
        "corruption_types": summary.get("corruption_types"),
        "by_corruption_grid": by_group,
        "by_corruption": by_corruption,
    }


def format_locality_markdown(report):
    lines = [
        "# Stage-3 Self-Corruption Locality Report",
        "",
        f"- manifest: `{report['manifest']}`",
        f"- row_count: {report['row_count']}",
    ]
    if report.get("prompt_count") is not None:
        lines.append(f"- prompt_count: {report['prompt_count']}")
    lines.extend([
        "",
        "| Corruption | Grid | N | Inside frac | In/out ratio | Center disp | Top1 | TopK | Median radius |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["by_corruption_grid"]:
        lines.append(
            "| {corruption_type} | {grid_size} | {row_count} | {inside} | {ratio} | {disp} | {top1} | {topk} | {radius} |".format(
                corruption_type=row["corruption_type"],
                grid_size=row["grid_size"],
                row_count=row["row_count"],
                inside=_fmt(row["mean_inside_energy_fraction"]),
                ratio=_fmt(row["mean_inside_outside_energy_ratio"]),
                disp=_fmt(row["mean_center_displacement_cells"]),
                top1=_fmt(row["top1_hit_rate"]),
                topk=_fmt(row["topk_hit_rate"]),
                radius=_fmt(row["median_effective_radius_cells"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value):
    return "null" if value is None else f"{float(value):.4f}"


def _normalise_path(value, project_root=None):
    path = Path(value)
    if project_root and path.is_absolute():
        try:
            return path.resolve().relative_to(Path(project_root).resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _row_token_grid_size(row, summary):
    corruption = row.get("corruption") or {}
    if corruption.get("token_grid_size") is not None:
        return int(corruption["token_grid_size"])
    for metric in row.get("metrics", []) or []:
        if metric.get("token_grid_size") is not None:
            return int(metric["token_grid_size"])
    if summary.get("token_grid_size") is not None:
        return int(summary["token_grid_size"])
    return None


def build_dataset_rows(rows, summary=None, project_root=None):
    summary = summary or {}
    dataset_rows = []
    for row in rows:
        corruption = row.get("corruption") or {}
        dataset_rows.append({
            "schema_version": DATASET_ROW_SCHEMA_VERSION,
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "clean_vq_ids_path": _normalise_path(row["clean_vq_ids_path"], project_root),
            "corrupted_vq_ids_path": _normalise_path(row["corrupted_vq_ids_path"], project_root),
            "clean_image": _normalise_path(row["clean_image"], project_root),
            "corrupted_image": _normalise_path(row["corrupted_image"], project_root),
            "corruption_indices": corruption.get("selected_indices", []),
            "corruption_type": _corruption_type(row),
            "token_id_space": corruption.get("token_id_space"),
            "selected_count": corruption.get("selected_count"),
            "changed_count": corruption.get("changed_count"),
            "coarse_labels_4x4": row.get("coarse_labels_4x4", []),
            "coarse_labels_8x8": row.get("coarse_labels_8x8", []),
            "coarse_labels_16x16": row.get("coarse_labels_16x16", []),
            "token_grid_size": _row_token_grid_size(row, summary),
            "image_size": summary.get("image_size"),
        })
    return dataset_rows


def build_self_corrupt_dataset(manifest_path, output_dir, summary_path=None, project_root=None):
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    summary = read_json(summary_path) if summary_path else {}
    source_rows = read_jsonl(manifest_path)
    dataset_rows = build_dataset_rows(source_rows, summary=summary, project_root=project_root)
    dataset_path = output_dir / "dataset.jsonl"
    manifest_output = output_dir / "dataset_manifest.json"
    write_jsonl(dataset_path, dataset_rows)
    corruption_types = sorted({_corruption_type(row) for row in source_rows})
    dataset_manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at_utc(),
        "source_manifest": str(manifest_path),
        "source_summary": str(summary_path) if summary_path else None,
        "dataset": str(dataset_path),
        "row_count": len(dataset_rows),
        "prompt_count": summary.get("prompt_count"),
        "corruption_types": corruption_types,
        "token_grid_size": summary.get("token_grid_size"),
        "image_size": summary.get("image_size"),
    }
    write_json(manifest_output, dataset_manifest)
    return dataset_manifest

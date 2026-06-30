"""Token-only Stage-3 repair dataset helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random

from ascr.corruption.vq_corruptor import corrupt_vq_ids_with_operator, token_indices_to_cell_labels


DATASET_ROW_SCHEMA = "ascr.stage3.token_repair_dataset.row.v2"
DATASET_MANIFEST_SCHEMA = "ascr.stage3.token_repair_dataset_manifest.v2"
CLEAN_ROW_SCHEMA = "ascr.stage3.clean_vq_token.row.v1"
ACTION_GRID_SIZE = 8
TOKEN_GRID_SIZE = 64
DEFAULT_MASK_SIZES = (1, 2, 4, 8)
DEFAULT_OPERATORS = ("random_replace", "local_shuffle", "neighbor_copy", "transplant")


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
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    return str(path)


def read_vq_ids(path):
    return [int(value) for value in json.loads(Path(path).read_text(encoding="utf-8"))]


def _normalise_path(path, project_root=None):
    raw = Path(path)
    if raw.is_absolute():
        return str(raw)
    if project_root:
        return str((Path(project_root) / raw).resolve())
    return str(raw)


def _load_clean_rows(paths, project_root=None):
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["source_clean_manifest"] = str(path)
            row["clean_vq_ids_path"] = _normalise_path(row["clean_vq_ids_path"], project_root=project_root)
            rows.append(row)
    if not rows:
        raise ValueError(f"No clean token rows found in manifests: {paths}")
    return rows


def _parse_int_list(values, default):
    if values is None:
        return list(default)
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    return [int(value) for value in values]


def _parse_str_list(values, default):
    if values is None:
        return list(default)
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    return [str(value).strip() for value in values if str(value).strip()]


def _choose_operator_and_size(rng, mask_sizes, operators):
    for _attempt in range(64):
        mask_size = rng.choice(mask_sizes)
        operator = rng.choice(operators)
        if not (operator == "local_shuffle" and int(mask_size) <= 1):
            return int(mask_size), operator
    return 2, "local_shuffle"


def _target_json(cells):
    labels = sorted({str(cell) for cell in cells})
    return {"cells": labels}


def _negative_row(clean_row, output_clean_path=None):
    sample_id = str(clean_row.get("sample_id") or f"clean_{clean_row.get('prompt_index', 'unknown')}")
    clean_path = str(output_clean_path or clean_row["clean_vq_ids_path"])
    return {
        "schema_version": DATASET_ROW_SCHEMA,
        "sample_id": f"{sample_id}:neg",
        "row_type": "negative",
        "split_group_id": sample_id,
        "source_clean_sample_id": sample_id,
        "prompt_index": clean_row.get("prompt_index"),
        "prompt": clean_row.get("prompt", ""),
        "clean_vq_ids_path": clean_path,
        "corrupted_vq_ids_path": clean_path,
        "target_schema": "repair_cells",
        "target_json": {"cells": []},
        "target_cells": [],
        "target_cells_8x8": [],
        "corruption_indices": [],
        "corruption_operator": "none",
        "corruption_mask_size": 0,
        "corruption_type": "clean_negative",
        "changed_count": 0,
        "token_grid_size": int(clean_row.get("token_grid_size") or TOKEN_GRID_SIZE),
        "action_grid_size": ACTION_GRID_SIZE,
        "image_size": int(clean_row.get("image_size") or 1024),
    }


def _positive_row(clean_row, clean_vq_ids, output_path, variant_index, rng, mask_sizes, operators, action_grid_size):
    sample_id = str(clean_row.get("sample_id") or f"clean_{clean_row.get('prompt_index', 'unknown')}")
    token_grid_size = int(clean_row.get("token_grid_size") or TOKEN_GRID_SIZE)
    for attempt in range(128):
        mask_size, operator = _choose_operator_and_size(rng, mask_sizes, operators)
        result = corrupt_vq_ids_with_operator(
            clean_vq_ids,
            token_grid_size=token_grid_size,
            mask_size=mask_size,
            operator=operator,
            seed=rng.randrange(0, 2**31 - 1),
        )
        if result.changed_count > 0:
            break
    else:
        raise RuntimeError(f"Could not create a changed corruption for clean sample {sample_id}")
    cells = token_indices_to_cell_labels(result.selected_indices, token_grid_size, action_grid_size)
    write_json(output_path, result.corrupted_vq_ids)
    return {
        "schema_version": DATASET_ROW_SCHEMA,
        "sample_id": f"{sample_id}:pos{variant_index:03d}",
        "row_type": "positive",
        "split_group_id": sample_id,
        "source_clean_sample_id": sample_id,
        "prompt_index": clean_row.get("prompt_index"),
        "prompt": clean_row.get("prompt", ""),
        "clean_vq_ids_path": str(clean_row["clean_vq_ids_path"]),
        "corrupted_vq_ids_path": str(output_path),
        "target_schema": "repair_cells",
        "target_json": _target_json(cells),
        "target_cells": cells,
        "target_cells_8x8": cells,
        "corruption_indices": list(result.selected_indices),
        "corruption_source_indices": list(result.source_indices or []),
        "corruption_source_mode": result.source_mode,
        "corruption_operator": result.operator,
        "corruption_mask_size": result.mask_size,
        "corruption_type": result.corruption_type,
        "changed_count": result.changed_count,
        "token_grid_size": token_grid_size,
        "action_grid_size": int(action_grid_size),
        "image_size": int(clean_row.get("image_size") or 1024),
    }


def build_token_repair_dataset(
    clean_manifests,
    output_dir,
    positive_rows=30000,
    negative_rows=10000,
    variants_per_clean=3,
    mask_sizes=None,
    operators=None,
    action_grid_size=ACTION_GRID_SIZE,
    seed=0,
    project_root=None,
):
    clean_rows = _load_clean_rows(clean_manifests, project_root=project_root)
    if len(clean_rows) < int(negative_rows):
        raise ValueError(
            f"Need at least {negative_rows} clean rows for unique negatives, got {len(clean_rows)}. "
            "Generate more clean VQ token samples or lower --negative-rows."
        )
    rng = random.Random(int(seed))
    clean_rows = list(clean_rows)
    rng.shuffle(clean_rows)
    mask_sizes = _parse_int_list(mask_sizes, DEFAULT_MASK_SIZES)
    operators = _parse_str_list(operators, DEFAULT_OPERATORS)
    output_dir = Path(output_dir)
    token_dir = output_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)

    selected_clean = clean_rows[: int(negative_rows)]
    rows = [_negative_row(row) for row in selected_clean]
    positives = []
    variant_counts = {str(row.get("sample_id") or index): 0 for index, row in enumerate(selected_clean)}
    clean_index = 0
    while len(positives) < int(positive_rows):
        row = selected_clean[clean_index % len(selected_clean)]
        clean_sample_id = str(row.get("sample_id") or clean_index)
        variant_index = variant_counts.get(clean_sample_id, 0)
        variant_counts[clean_sample_id] = variant_index + 1
        clean_vq_ids = read_vq_ids(row["clean_vq_ids_path"])
        output_path = token_dir / f"{clean_sample_id.replace(':', '_')}_pos{variant_index:03d}_corrupted_vq_ids.json"
        positives.append(
            _positive_row(
                row,
                clean_vq_ids,
                output_path,
                variant_index=variant_index,
                rng=rng,
                mask_sizes=mask_sizes,
                operators=operators,
                action_grid_size=action_grid_size,
            )
        )
        clean_index += 1
        if int(variants_per_clean) > 0 and clean_index >= len(selected_clean) * int(variants_per_clean):
            clean_index = 0
    rows.extend(positives)
    rows.sort(key=lambda row: (str(row.get("source_clean_sample_id")), str(row.get("row_type")), str(row.get("sample_id"))))

    dataset_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"
    write_jsonl(dataset_path, rows)
    op_counts = {}
    size_counts = {}
    cell_counts = {}
    for row in positives:
        op_counts[row["corruption_operator"]] = op_counts.get(row["corruption_operator"], 0) + 1
        size = str(row["corruption_mask_size"])
        size_counts[size] = size_counts.get(size, 0) + 1
        for cell in row["target_cells_8x8"]:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "created_at_utc": created_at_utc(),
        "clean_manifests": [str(path) for path in clean_manifests],
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "positive_rows": len(positives),
        "negative_rows": len(rows) - len(positives),
        "clean_sample_count": len(selected_clean),
        "variants_per_clean_requested": int(variants_per_clean),
        "mask_sizes": mask_sizes,
        "operators": operators,
        "action_grid_size": int(action_grid_size),
        "target_schema": "repair_cells",
        "operator_counts": op_counts,
        "mask_size_counts": size_counts,
        "target_cell_counts": dict(sorted(cell_counts.items())),
        "seed": int(seed),
    }
    write_json(manifest_path, manifest)
    return manifest

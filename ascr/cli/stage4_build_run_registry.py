"""Build a compact registry of Stage-4 SFT, LoRA, and probe artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifact_id(path, roots):
    path = Path(path)
    for root in roots:
        try:
            return str(path.resolve().relative_to(Path(root).resolve())).replace("\\", "/")
        except Exception:
            continue
    return str(path).replace("\\", "/")


def _scan_json_files(roots, filename):
    files = []
    for root in roots:
        root = Path(root)
        if root.exists():
            files.extend(sorted(root.rglob(filename)))
    return files


def _path_keys(path):
    candidate = Path(path)
    keys = {str(candidate)}
    try:
        keys.add(str(candidate.resolve()))
    except Exception:
        pass
    return keys


def _row_from_training_manifest(path, payload, roots):
    return {
        "kind": "lora_adapter",
        "artifact_id": _artifact_id(path.parent, roots),
        "path": str(path.parent),
        "manifest_path": str(path),
        "data_jsonl": payload.get("data_jsonl"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "repo_path": payload.get("repo_path"),
        "row_count": payload.get("row_count"),
        "epochs": payload.get("epochs"),
        "final_loss": payload.get("final_loss"),
        "optimizer": payload.get("optimizer"),
        "image_size": payload.get("image_size"),
        "max_seq_len": payload.get("max_seq_len"),
        "target_modules": payload.get("target_modules"),
        "torch_dtype": payload.get("torch_dtype"),
        "gradient_checkpointing": payload.get("gradient_checkpointing"),
        "gradient_checkpointing_backend": (payload.get("gradient_checkpointing_report") or {}).get("backend"),
        "wrapped_module_count": (payload.get("gradient_checkpointing_report") or {}).get("wrapped_module_count"),
    }


def _row_from_sft_manifest(path, payload, roots):
    return {
        "kind": "sft_dataset",
        "artifact_id": _artifact_id(path.parent, roots),
        "path": str(path.parent),
        "manifest_path": str(path),
        "dataset": payload.get("dataset"),
        "input_mode": payload.get("input_mode"),
        "input_modes": payload.get("input_modes"),
        "target_schema": payload.get("target_schema"),
        "grid_size": payload.get("grid_size"),
        "source_row_count": payload.get("source_row_count"),
        "train_rows": payload.get("train_rows"),
        "eval_rows": payload.get("eval_rows"),
        "missing_required_inputs": payload.get("missing_required_inputs"),
        "train_sft_examples": payload.get("train_sft_examples"),
        "eval_sft_examples": payload.get("eval_sft_examples"),
    }


def _row_from_probe_summary(path, payload, roots):
    metrics = payload.get("metrics") or {}
    return {
        "kind": "probe_summary",
        "artifact_id": _artifact_id(path.parent, roots),
        "path": str(path.parent),
        "summary_path": str(path),
        "dataset": payload.get("dataset"),
        "row_count": payload.get("row_count"),
        "grid_size": payload.get("grid_size"),
        "input_mode": payload.get("input_mode"),
        "target_schema": payload.get("target_schema"),
        "prompt_variant": payload.get("prompt_variant"),
        "lora_path": payload.get("lora_path"),
        "max_new_tokens": payload.get("max_new_tokens"),
        "answer_steps": payload.get("answer_steps"),
        "answer_block_length": payload.get("answer_block_length"),
        "answer_temperature": payload.get("answer_temperature"),
        "answer_cfg_scale": payload.get("answer_cfg_scale"),
        "parse_rate": payload.get("parse_rate"),
        "parsed_count": payload.get("parsed_count"),
        "malformed_count": payload.get("malformed_count"),
        "call_error_count": payload.get("call_error_count"),
        "hit_any_rate": metrics.get("hit_any_rate"),
        "mean_f1_at_k": metrics.get("mean_f1_at_k"),
        "mean_iou": metrics.get("mean_iou"),
        "mean_distance_to_target_cells": metrics.get("mean_distance_to_target_cells"),
    }


def build_registry(roots):
    roots = [Path(root) for root in roots]
    rows = []
    adapter_by_path = {}
    for path in _scan_json_files(roots, "training_manifest.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        row = _row_from_training_manifest(path, payload, roots)
        rows.append(row)
        for key in _path_keys(row["path"]):
            adapter_by_path[key] = row
    for path in _scan_json_files(roots, "manifest.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") == "ascr.stage4.mmu_lora_sft_manifest.v1":
            rows.append(_row_from_sft_manifest(path, payload, roots))
    for path in _scan_json_files(roots, "summary.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") == "ascr.stage4.mmu_localization_probe.summary.v1":
            row = _row_from_probe_summary(path, payload, roots)
            lora_path = row.get("lora_path")
            if lora_path:
                adapter = next((adapter_by_path[key] for key in _path_keys(lora_path) if key in adapter_by_path), None)
                if adapter:
                    row["adapter_artifact_id"] = adapter["artifact_id"]
            rows.append(row)
    rows.sort(key=lambda item: (str(item.get("kind")), str(item.get("artifact_id"))))
    return {
        "schema_version": "ascr.stage4.run_registry.v1",
        "created_at_utc": _created_at(),
        "roots": [str(root) for root in roots],
        "row_count": len(rows),
        "rows": rows,
    }


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def write_outputs(output_dir, registry):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage4_run_registry.json"
    markdown_path = output_dir / "stage4_run_registry.md"
    json_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-4 Run Registry",
        "",
        f"Rows: {registry['row_count']}",
        "",
        "| Kind | Artifact | Grid | Input | Parse | Hit any | Loss | GC backend | Path |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in registry["rows"]:
        lines.append(
            "| {kind} | {artifact} | {grid} | {input_mode} | {parse} | {hit} | {loss} | {gc} | `{path}` |".format(
                kind=_fmt(row.get("kind")),
                artifact=_fmt(row.get("artifact_id")),
                grid=_fmt(row.get("grid_size")),
                input_mode=_fmt(row.get("input_mode") or row.get("input_modes")),
                parse=_fmt(row.get("parse_rate")),
                hit=_fmt(row.get("hit_any_rate")),
                loss=_fmt(row.get("final_loss")),
                gc=_fmt(row.get("gradient_checkpointing_backend")),
                path=_fmt(row.get("path")),
            )
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"registry_json": str(json_path), "registry_md": str(markdown_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Build a Stage-4 run registry from output roots.")
    parser.add_argument("--roots", nargs="+", required=True, help="Output roots to scan.")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    registry = build_registry(args.roots)
    outputs = write_outputs(args.output_dir, registry)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

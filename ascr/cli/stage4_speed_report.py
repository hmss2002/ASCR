"""Summarize Stage-4 LoRA training speed from training manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path):
    path = Path(path)
    if path.is_dir():
        path = path / "training_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def summarize_training_manifest(path, label=None):
    manifest_path, payload = _read_json(path)
    epoch_summaries = payload.get("epoch_summaries") or []
    epoch_elapsed = [item.get("elapsed_s") for item in epoch_summaries]
    epoch_samples_s = [
        item.get("approx_global_samples_per_s", item.get("samples_per_s"))
        for item in epoch_summaries
    ]
    validated_epochs = [item for item in epoch_summaries if item.get("validated")]
    cache_report = payload.get("cache_report") or {}
    total_elapsed = sum(float(value) for value in epoch_elapsed if value is not None)
    completed_epochs = int(payload.get("completed_epochs") or len(epoch_summaries) or 0)
    row = {
        "label": str(label or manifest_path.parent.name),
        "manifest_path": str(manifest_path),
        "output_dir": str(payload.get("output_dir") or manifest_path.parent),
        "row_count": payload.get("row_count"),
        "completed_epochs": completed_epochs,
        "requested_epochs": payload.get("epochs"),
        "world_size": payload.get("world_size"),
        "optimizer": payload.get("optimizer"),
        "gradient_checkpointing": bool(payload.get("gradient_checkpointing")),
        "gradient_checkpointing_backend": (payload.get("gradient_checkpointing_report") or {}).get("backend"),
        "validation_every_epochs": payload.get("validation_every_epochs"),
        "validated_epoch_count": len(validated_epochs),
        "cache_report": cache_report,
        "image_token_cache_size": cache_report.get("image_token_cache_size"),
        "text_token_cache_size": cache_report.get("text_token_cache_size"),
        "torch_dtype": payload.get("torch_dtype"),
        "max_seq_len": payload.get("max_seq_len"),
        "lora_r": payload.get("lora_r"),
        "total_epoch_elapsed_s": total_elapsed if epoch_summaries else None,
        "mean_epoch_elapsed_s": _mean(epoch_elapsed),
        "mean_global_samples_per_s": _mean(epoch_samples_s),
        "best_global_samples_per_s": max([float(value) for value in epoch_samples_s if value is not None], default=None),
        "final_loss": payload.get("final_loss"),
        "best_val_loss": payload.get("best_val_loss"),
    }
    return row


def build_speed_report(manifests, labels=None, baseline_label=None):
    labels = list(labels or [])
    rows = []
    for index, manifest in enumerate(manifests):
        label = labels[index] if index < len(labels) else None
        rows.append(summarize_training_manifest(manifest, label=label))
    baseline = None
    if baseline_label:
        baseline = next((row for row in rows if row["label"] == baseline_label), None)
    if baseline is None and rows:
        baseline = rows[0]
    baseline_speed = baseline.get("mean_global_samples_per_s") if baseline else None
    for row in rows:
        speed = row.get("mean_global_samples_per_s")
        if baseline_speed and speed:
            row["speedup_vs_baseline"] = float(speed) / float(baseline_speed)
        else:
            row["speedup_vs_baseline"] = None
    return {
        "schema_version": "ascr.stage4.speed_report.v1",
        "created_at_utc": _created_at(),
        "baseline_label": baseline.get("label") if baseline else None,
        "row_count": len(rows),
        "rows": rows,
    }


def write_outputs(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage4_speed_report.json"
    md_path = output_dir / "stage4_speed_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-4 Speed Report",
        "",
        f"Baseline: `{report.get('baseline_label') or ''}`",
        "",
        "| Label | Optimizer | GC | Val every | Cache | World | Epochs | Mean epoch s | Mean samples/s | Speedup | Manifest |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        def fmt(value):
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.4g}"
            return str(value)

        lines.append(
            "| {label} | {optimizer} | {gc} | {val_every} | {cache} | {world} | {epochs} | {epoch_s} | {samples_s} | {speedup} | `{manifest}` |".format(
                label=fmt(row.get("label")),
                optimizer=fmt(row.get("optimizer")),
                gc=fmt(row.get("gradient_checkpointing")),
                val_every=fmt(row.get("validation_every_epochs")),
                cache=fmt(
                    "img={image}, text={text}".format(
                        image=row.get("image_token_cache_size") or "",
                        text=row.get("text_token_cache_size") or "",
                    )
                ),
                world=fmt(row.get("world_size")),
                epochs=fmt(row.get("completed_epochs")),
                epoch_s=fmt(row.get("mean_epoch_elapsed_s")),
                samples_s=fmt(row.get("mean_global_samples_per_s")),
                speedup=fmt(row.get("speedup_vs_baseline")),
                manifest=fmt(row.get("manifest_path")),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"speed_report_json": str(json_path), "speed_report_md": str(md_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize Stage-4 LoRA training speed.")
    parser.add_argument("--manifests", nargs="+", required=True, help="training_manifest.json files or adapter dirs.")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = build_speed_report(args.manifests, labels=args.labels, baseline_label=args.baseline_label)
    if args.output_dir:
        outputs = write_outputs(args.output_dir, report)
        print(json.dumps(outputs, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

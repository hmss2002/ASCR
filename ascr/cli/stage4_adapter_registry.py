"""Scan Stage-4 LoRA adapters and nearby evaluation metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _nearest_probe_summary(adapter_dir):
    parent = adapter_dir.parent
    candidates = sorted(parent.glob("probe*/summary.json"))
    return candidates[-1] if candidates else None


def scan_adapters(root):
    rows = []
    for adapter_file in sorted(Path(root).rglob("adapter_model.safetensors")):
        adapter_dir = adapter_file.parent
        train = _read_json(adapter_dir / "training_manifest.json")
        probe_path = _nearest_probe_summary(adapter_dir)
        probe = _read_json(probe_path) if probe_path else {}
        rows.append({
            "adapter_dir": str(adapter_dir),
            "adapter_model": str(adapter_file),
            "grid_size": probe.get("grid_size") or train.get("grid_size"),
            "prompt_variant": probe.get("prompt_variant") or train.get("prompt_variant"),
            "training_loss": train.get("final_loss"),
            "eval_parse_rate": probe.get("parse_rate"),
            "eval_hit_any_rate": (probe.get("metrics") or {}).get("hit_any_rate"),
            "image_size": train.get("image_size"),
            "gradient_checkpointing": train.get("gradient_checkpointing"),
            "probe_summary": str(probe_path) if probe_path else None,
        })
    return {
        "schema_version": "ascr.stage4.adapter_registry.v1",
        "created_at_utc": created_at_utc(),
        "scan_root": str(root),
        "adapter_count": len(rows),
        "adapters": rows,
    }


def write_registry(path, registry):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def build_parser():
    parser = argparse.ArgumentParser(description="Scan Stage-4 LoRA adapter metadata.")
    parser.add_argument("--scan", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    registry = scan_adapters(args.scan)
    print(json.dumps({"output": write_registry(args.output, registry), "adapter_count": registry["adapter_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


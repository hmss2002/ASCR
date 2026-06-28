"""Run Stage-4 grid training/eval batches inside one allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path


TRAIN_CONFIGS = {
    4: "configs/stage4/self_corrupt/mmu_lora_train_hard64_grid4_vq_tokens_l40s_1024px_gc_adam8bit.yaml",
    8: "configs/stage4/self_corrupt/mmu_lora_train_hard64_grid8_vq_tokens_l40s_1024px_gc_adam8bit.yaml",
    16: "configs/stage4/self_corrupt/mmu_lora_train_hard64_grid16_vq_tokens_l40s_1024px_gc_adam8bit.yaml",
}
PROBE_CONFIGS = {
    4: "configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid4_vq_tokens_l40s_1024px_gc.yaml",
    8: "configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid8_vq_tokens_l40s_1024px_gc.yaml",
    16: "configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid16_vq_tokens_l40s_1024px_gc.yaml",
}


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_grids(value):
    return [int(part) for part in str(value).replace(",", " ").split() if part.strip()]


def run_command(command, dry_run=False):
    if dry_run:
        return {"command": command, "returncode": None, "dry_run": True}
    completed = subprocess.run(command, check=False)
    return {"command": command, "returncode": completed.returncode, "dry_run": False}


def build_parser():
    parser = argparse.ArgumentParser(description="Train/probe multiple Stage-4 grids in one process allocation.")
    parser.add_argument("--grids", default="4,8,16")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/stage4_self_corrupt/batch_train")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    python_bin = os.environ.get("PYTHON_BIN") or sys.executable or "python"
    results = []
    for grid in parse_grids(args.grids):
        train_cmd = [python_bin, "-m", "ascr.cli.stage4_train_mmu_lora", "--config", TRAIN_CONFIGS[grid]]
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if args.limit is not None:
            train_cmd.extend(["--limit", str(args.limit)])
        results.append({"grid": grid, "kind": "train", **run_command(train_cmd, dry_run=args.dry_run)})
        if results[-1].get("returncode") not in {None, 0}:
            break
        if args.run_probe:
            probe_cmd = [python_bin, "-m", "ascr.cli.stage4_mmu_localization_probe", "--config", PROBE_CONFIGS[grid]]
            results.append({"grid": grid, "kind": "probe", **run_command(probe_cmd, dry_run=args.dry_run)})
            if results[-1].get("returncode") not in {None, 0}:
                break
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "ascr.stage4.batch_train.v1", "created_at_utc": created_at_utc(), "results": results}
    path = Path(args.output_dir) / "batch_train_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "results": results}, indent=2, sort_keys=True))
    return 0 if all(item.get("returncode") in {None, 0} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

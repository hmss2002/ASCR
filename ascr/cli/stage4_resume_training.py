"""Resume-or-skip wrapper for Stage-4 MMU LoRA training jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from ascr.core.config import load_config


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _python_bin():
    return os.environ.get("PYTHON_BIN") or sys.executable or "python"


def _is_complete(manifest_path):
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest.get("final_loss") is not None and int(manifest.get("epochs") or 0) > 0


def _write_decision(output_dir, payload):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resume_decision.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _latest_checkpoint(output_dir):
    root = Path(output_dir) / "checkpoints"
    if not root.exists():
        return None
    candidates = []
    for path in root.glob("epoch_*"):
        if path.is_dir() and (path / "adapter_config.json").exists():
            candidates.append(path)
    if not candidates:
        return None
    return str(sorted(candidates)[-1])


def build_parser():
    parser = argparse.ArgumentParser(description="Skip completed Stage-4 LoRA runs or relaunch interrupted ones.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.get("output_dir") or "outputs/stage4_self_corrupt/resume_training")
    manifest_path = output_dir / "training_manifest.json"
    command = [
        _python_bin(),
        "-m",
        "ascr.cli.stage4_train_mmu_lora_ddp" if args.ddp else "ascr.cli.stage4_train_mmu_lora",
        "--config",
        args.config,
    ]
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    latest_checkpoint = _latest_checkpoint(output_dir)
    if latest_checkpoint and not args.force:
        command.extend(["--resume-from-adapter", latest_checkpoint])
    if args.checkpoint_every_epochs is not None:
        command.extend(["--checkpoint-every-epochs", str(args.checkpoint_every_epochs)])

    skipped = _is_complete(manifest_path) and not args.force
    decision = {
        "schema_version": "ascr.stage4.resume_training.v1",
        "created_at_utc": created_at_utc(),
        "config": str(args.config),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "ddp": bool(args.ddp),
        "force": bool(args.force),
        "dry_run": bool(args.dry_run),
        "skipped": bool(skipped),
        "latest_checkpoint": latest_checkpoint,
        "command": command,
        "returncode": None,
        "note": "Complete manifests are skipped. If checkpoint adapters exist, the latest checkpoint is passed as --resume-from-adapter; otherwise interrupted runs relaunch from config.",
    }
    if skipped or args.dry_run:
        decision["decision_path"] = _write_decision(output_dir, decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    completed = subprocess.run(command, check=False)
    decision["returncode"] = completed.returncode
    decision["decision_path"] = _write_decision(output_dir, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

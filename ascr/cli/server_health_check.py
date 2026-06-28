"""ASCR server-side health checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def _check_path(label, path):
    candidate = Path(path)
    return {"label": label, "path": str(candidate), "ok": candidate.exists()}


def _torch_cuda():
    try:
        import torch

        return {"ok": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count())}
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _recent_log_errors(log_dir):
    errors = []
    if not Path(log_dir).exists():
        return errors
    needles = ("outofmemory", "cuda out of memory", "traceback", "error", "qosmax")
    for path in sorted(Path(log_dir).glob("*.err"), key=lambda item: item.stat().st_mtime, reverse=True)[:20]:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if any(needle in text for needle in needles):
            errors.append(str(path))
    return errors


def health_check(args):
    disk = shutil.disk_usage(args.project_root)
    checks = [
        _check_path("project_root", args.project_root),
        _check_path("venv_python", Path(args.env) / "bin" / "python"),
        _check_path("lumina_repo", args.lumina_repo),
        _check_path("lumina_model", args.lumina_model),
    ]
    squeue = subprocess.run(["squeue", "--version"], check=False, text=True, capture_output=True) if shutil.which("squeue") else None
    payload = {
        "schema_version": "ascr.server_health_check.v1",
        "checks": checks,
        "all_paths_ok": all(check["ok"] for check in checks),
        "torch_cuda": _torch_cuda(),
        "slurm_available": bool(squeue and squeue.returncode == 0),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "recent_error_logs": _recent_log_errors(args.log_dir),
        "env": {
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            "LUMINA_REPO": os.environ.get("LUMINA_REPO"),
            "LUMINA_MODEL_PATH": os.environ.get("LUMINA_MODEL_PATH"),
        },
    }
    payload["ok"] = payload["all_paths_ok"] and payload["disk_free_gb"] > float(args.min_free_gb)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Run ASCR server health checks.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--env", default=".venv-lumina")
    parser.add_argument("--lumina-repo", default=os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO"))
    parser.add_argument("--lumina-model", default=os.environ.get("LUMINA_MODEL_PATH", "models/lumina-dimoo"))
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    return parser


def main(argv=None):
    payload = health_check(build_parser().parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


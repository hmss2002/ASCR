from pathlib import Path
from datetime import datetime
import json
import os
import platform
import subprocess
import sys


def current_git_commit(project_root):
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(project_root), check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def runtime_manifest(project_root):
    env_keys = [
        "ASCR_ENV_QWEN",
        "ASCR_ENV_LUMINA",
        "ASCR_ENV_MMADA",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "OUT_ROOT",
        "PROMPT_LIMIT",
        "NUM_WORKERS",
        "NUM_PAIRS",
        "NODE_INDEX",
        "NODE_COUNT",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
    ]
    env = {}
    for key in env_keys:
        if key in os.environ:
            env[key] = os.environ.get(key)
    return {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "project_root": str(Path(project_root).resolve()),
        "git_commit": current_git_commit(project_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "env": env,
    }


class RunArtifacts:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "iterations").mkdir(exist_ok=True)

    @classmethod
    def create(cls, output_dir, run_name="stage1"):
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        pid = os.getpid()
        return cls(Path(output_dir) / f"{run_name}-{timestamp}-{pid}")

    def iteration_dir(self, iteration):
        path = self.root / "iterations" / f"{iteration:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path, payload):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write(chr(10))
        return path

    def write_text(self, relative_path, text):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

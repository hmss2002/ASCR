from pathlib import Path
from datetime import datetime
import json
import os
import subprocess


def current_git_commit(project_root):
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(project_root), check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except Exception:
        return "unknown"


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

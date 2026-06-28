"""Print a compact server/Slurm dashboard for ASCR runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _run(command):
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        return {"ok": completed.returncode == 0, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except FileNotFoundError as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc)}


def build_dashboard(log_dir="logs", output_root="outputs"):
    queue = _run(["squeue", "-u", _run(["whoami"])["stdout"] or "", "-o", "%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R"])
    sacct = _run(["sacct", "-n", "-S", "now-24hours", "-o", "JobID,JobName%28,State,Elapsed"])
    recent_logs = []
    for path in sorted(Path(log_dir).glob("*.out"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)[:10]:
        recent_logs.append(str(path))
    output_dirs = [str(path) for path in sorted(Path(output_root).glob("*"))[:20]] if Path(output_root).exists() else []
    return {
        "schema_version": "ascr.server_dashboard.v1",
        "queue_ok": queue["ok"],
        "queue": queue["stdout"] or queue["stderr"],
        "recent_jobs_ok": sacct["ok"],
        "recent_jobs": sacct["stdout"] or sacct["stderr"],
        "recent_logs": recent_logs,
        "output_dirs": output_dirs,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Print ASCR server queue and artifact dashboard.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dashboard = build_dashboard(log_dir=args.log_dir, output_root=args.output_root)
    if args.json:
        print(json.dumps(dashboard, indent=2, sort_keys=True))
    else:
        print("# ASCR Server Dashboard\n")
        print("## Queue\n")
        print(dashboard["queue"] or "(no queue data)")
        print("\n## Recent Jobs\n")
        print(dashboard["recent_jobs"] or "(no sacct data)")
        print("\n## Recent Logs")
        for path in dashboard["recent_logs"]:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Run lightweight ASCR checks that do not require model weights or GPUs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_CONFIG = "configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml"


def _run(cmd, cwd):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), check=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run ASCR local smoke checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config to inspect in preflight.")
    parser.add_argument("--output-dir", default="outputs/local_smoke", help="Dry-run output directory.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unittest discovery.")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip mock Stage-1 dry run.")
    parser.add_argument("--server", action="store_true", help="Use server preflight mode; requires torch/CUDA.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    commands = []
    if not args.skip_tests:
        commands.append([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    commands.append([sys.executable, "-m", "ascr.cli.run_stage1", "--help"])
    if not args.skip_dry_run:
        commands.append([
            sys.executable,
            "-m",
            "ascr.cli.run_stage1",
            "--dry-run",
            "--max-iterations",
            "1",
            "--output-dir",
            args.output_dir,
        ])
    commands.append([
        sys.executable,
        "-m",
        "ascr.cli.preflight",
        "--mode",
        "server" if args.server else "local",
        "--config",
        args.config,
        "--scan-secrets",
    ])

    for cmd in commands:
        completed = subprocess.run(cmd, cwd=str(root), env=env, check=False)
        if completed.returncode != 0:
            print(f"FAILED: {' '.join(cmd)}", file=sys.stderr)
            return completed.returncode
    print("ASCR smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

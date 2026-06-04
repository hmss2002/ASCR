#!/usr/bin/env python3
"""
Remove entries with empty `raw` field from all 9 bench3 checkpoint files.
This allows the eval scripts to re-query those items in the next run.

Usage:
    python scripts/purge_empty_checkpoints.py --eval-dir outputs/bench3_eval
"""

import argparse
import json
from pathlib import Path


TASKS = [
    "dpg_showo", "dpg_ascr", "dpg_bagel",
    "dsg_showo", "dsg_ascr", "dsg_bagel",
    "genai_showo", "genai_ascr", "genai_bagel",
]


def purge_task(task_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Purge empty-raw entries from checkpoint.jsonl. Returns (kept, removed)."""
    ckpt = task_dir / "checkpoint.jsonl"
    if not ckpt.exists():
        print(f"  {task_dir.name}: no checkpoint file, skipping")
        return 0, 0

    kept, removed = [], 0
    with ckpt.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("raw", "") == "":
                removed += 1
            else:
                kept.append(line)

    print(f"  {task_dir.name}: keeping {len(kept)}, removing {removed} empty-raw entries")
    if not dry_run and removed > 0:
        backup = task_dir / "checkpoint.jsonl.bak"
        ckpt.rename(backup)
        with ckpt.open("w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        print(f"    -> backup saved to {backup.name}")

    return len(kept), removed


def main():
    parser = argparse.ArgumentParser(description="Purge empty-raw entries from bench3 checkpoints.")
    parser.add_argument("--eval-dir", default="outputs/bench3_eval",
                        help="Directory containing per-task subdirectories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without modifying files")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    total_kept = total_removed = 0
    for task in TASKS:
        task_dir = eval_dir / task
        if not task_dir.exists():
            print(f"  {task}: directory not found, skipping")
            continue
        k, r = purge_task(task_dir, dry_run=args.dry_run)
        total_kept += k
        total_removed += r

    print(f"\nTotal: {total_kept} kept, {total_removed} removed")
    if args.dry_run:
        print("(DRY RUN — no files were modified)")


if __name__ == "__main__":
    main()

"""Build or execute a Stage-6 multi-arm benchmark plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


KNOWN_ARMS = ("direct", "stage1_qwen", "stage3_selector", "stage4_mmu_lora")


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _command_for_arm(arm, prompts, limit, output_dir, lora_path=None, mock=False):
    arm_dir = Path(output_dir) / arm
    python_bin = os.environ.get("PYTHON_BIN") or sys.executable or "python"
    if arm in {"direct", "stage4_mmu_lora"}:
        cmd = [
            python_bin, "-m", "ascr.cli.stage6_transfer_probe",
            "--prompts", str(prompts),
            "--output-dir", str(arm_dir),
        ]
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if arm == "stage4_mmu_lora" and lora_path:
            cmd.extend(["--lora-path", str(lora_path)])
        if mock:
            cmd.append("--mock")
        return cmd
    return [
        python_bin, "-m", "ascr.cli.stage6_multi_arm_benchmark",
        "--write-placeholder-arm", arm,
        "--prompts", str(prompts),
        "--output-dir", str(arm_dir),
        *(["--limit", str(limit)] if limit is not None else []),
    ]


def _read_prompts(path, limit=None):
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            prompts.append(text)
        if limit is not None and len(prompts) >= int(limit):
            break
    return prompts


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    return str(path)


def write_placeholder_arm(arm, prompts, output_dir, limit=None):
    prompt_rows = _read_prompts(prompts, limit=limit)
    rows = [
        {
            "schema_version": "ascr.stage6.multi_arm.placeholder_row.v1",
            "arm": arm,
            "sample_index": index,
            "prompt": prompt,
            "status": "placeholder",
            "note": "Placeholder arm manifest; plug in the mature runner when ready.",
        }
        for index, prompt in enumerate(prompt_rows)
    ]
    manifest = _write_jsonl(Path(output_dir) / "manifest.jsonl", rows)
    return _write_json(Path(output_dir) / "summary.json", {
        "schema_version": "ascr.stage6.multi_arm.placeholder_summary.v1",
        "created_at_utc": created_at_utc(),
        "arm": arm,
        "row_count": len(rows),
        "manifest": manifest,
    })


def build_plan(args):
    arms = args.arms or list(KNOWN_ARMS)
    unknown = [arm for arm in arms if arm not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"Unknown benchmark arms: {unknown}")
    plan = {
        "schema_version": "ascr.stage6.multi_arm_benchmark.plan.v1",
        "created_at_utc": created_at_utc(),
        "arms": arms,
        "prompts": args.prompts,
        "limit": args.limit,
        "output_dir": args.output_dir,
        "commands": [
            {"arm": arm, "command": _command_for_arm(arm, args.prompts, args.limit, args.output_dir, lora_path=args.lora_path, mock=args.mock)}
            for arm in arms
        ],
    }
    return plan


def run_plan(plan):
    results = []
    for item in plan["commands"]:
        completed = subprocess.run(item["command"], check=False)
        results.append({"arm": item["arm"], "returncode": completed.returncode})
        if completed.returncode != 0:
            break
    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Plan or execute a Stage-6 multi-arm benchmark.")
    parser.add_argument("--arms", nargs="+", choices=KNOWN_ARMS, default=None)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--write-placeholder-arm", choices=KNOWN_ARMS, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.write_placeholder_arm:
        print(write_placeholder_arm(args.write_placeholder_arm, args.prompts, args.output_dir, limit=args.limit))
        return 0
    plan = build_plan(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _write_json(Path(args.output_dir) / "multi_arm_plan.json", plan)
    if args.execute:
        plan["results"] = run_plan(plan)
        _write_json(Path(args.output_dir) / "multi_arm_results.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

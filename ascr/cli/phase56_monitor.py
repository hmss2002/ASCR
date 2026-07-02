"""Monitor ASCR Phase-5/6 Slurm jobs and output artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_command(args):
    completed = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def job_status(job_id):
    if not job_id:
        return {"job_id": None, "state": "not_configured"}
    code, stdout, stderr = run_command([
        "squeue",
        "-h",
        "-j",
        str(job_id),
        "-o",
        "%i|%T|%R|%S|%E",
    ])
    if code == 0 and stdout:
        line = stdout.splitlines()[0]
        parts = (line.split("|") + ["", "", "", "", ""])[:5]
        return {
            "job_id": str(job_id),
            "source": "squeue",
            "state": parts[1],
            "reason": parts[2],
            "start_time": parts[3],
            "dependency": parts[4],
        }
    code, stdout, stderr = run_command([
        "sacct",
        "-n",
        "-P",
        "-j",
        str(job_id),
        "--format=JobIDRaw,State,ExitCode,Elapsed",
    ])
    rows = []
    if code == 0 and stdout:
        for line in stdout.splitlines():
            job_raw, state, exit_code, elapsed = (line.split("|") + ["", "", "", ""])[:4]
            rows.append({
                "job_id_raw": job_raw,
                "state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
            })
    if rows:
        return {
            "job_id": str(job_id),
            "source": "sacct",
            "state": rows[0]["state"],
            "rows": rows,
        }
    return {
        "job_id": str(job_id),
        "source": "unknown",
        "state": "unknown",
        "error": stderr,
    }


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path):
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def artifact_status(args):
    smoke_trace = read_json(Path(args.stage5_smoke_dir) / "trace.json")
    multi_summary = read_json(Path(args.stage5_multi_dir) / "summary.json")
    transfer_summary = read_json(Path(args.stage6_transfer_dir) / "summary.json")
    multi_manifest = Path(args.stage5_multi_dir) / "manifest.jsonl"
    multi_comparison = Path(args.stage5_multi_dir) / "comparison" / "stage5_loop_comparison.json"
    status = {
        "stage5_smoke": {
            "dir": args.stage5_smoke_dir,
            "trace_exists": smoke_trace is not None,
            "grid_size": smoke_trace.get("grid_size") if smoke_trace else None,
            "target_schema": smoke_trace.get("target_schema") if smoke_trace else None,
            "reopen_changed": smoke_trace.get("reopen_changed") if smoke_trace else None,
            "lora_cells": smoke_trace.get("lora_cells") if smoke_trace else None,
            "target_cells": smoke_trace.get("target_cells") if smoke_trace else None,
        },
        "stage5_multi_prompt": {
            "dir": args.stage5_multi_dir,
            "summary_exists": multi_summary is not None,
            "manifest_exists": multi_manifest.exists(),
            "manifest_rows": count_jsonl(multi_manifest),
            "comparison_exists": multi_comparison.exists(),
            "ok_count": multi_summary.get("ok_count") if multi_summary else None,
            "error_count": multi_summary.get("error_count") if multi_summary else None,
        },
        "stage6_transfer": {
            "dir": args.stage6_transfer_dir,
            "summary_exists": transfer_summary is not None,
            "row_count": transfer_summary.get("row_count") if transfer_summary else None,
            "parsed_count": transfer_summary.get("parsed_count") if transfer_summary else None,
            "nonempty_count": transfer_summary.get("nonempty_count") if transfer_summary else None,
            "grid_size": transfer_summary.get("grid_size") if transfer_summary else None,
            "target_schema": transfer_summary.get("target_schema") if transfer_summary else None,
        },
    }
    return status


def recommendation(jobs, artifacts):
    smoke = jobs["stage5_smoke"]["state"]
    multi = jobs["stage5_multi_prompt"]["state"]
    transfer = jobs["stage6_transfer"]["state"]
    if not artifacts["stage5_smoke"]["trace_exists"]:
        return "wait_for_stage5_smoke"
    if artifacts["stage5_smoke"]["target_schema"] != "repair_cells" or artifacts["stage5_smoke"]["grid_size"] != 8:
        return "inspect_stage5_smoke_schema_mismatch"
    if not artifacts["stage5_multi_prompt"]["manifest_exists"]:
        if multi in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            return "wait_for_stage5_multi_prompt"
        if smoke in {"FAILED", "TIMEOUT", "CANCELLED"}:
            return "inspect_stage5_smoke_failure"
        return "submit_or_requeue_stage5_multi_prompt"
    if artifacts["stage5_multi_prompt"]["error_count"] not in {None, 0}:
        return "inspect_stage5_multi_prompt_errors"
    if not artifacts["stage6_transfer"]["summary_exists"]:
        if transfer in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            return "wait_for_stage6_transfer"
        return "submit_or_requeue_stage6_transfer"
    if artifacts["stage6_transfer"]["target_schema"] != "repair_cells" or artifacts["stage6_transfer"]["grid_size"] != 8:
        return "inspect_stage6_schema_mismatch"
    return "phase5_phase6_smoke_chain_complete"


def build_report(args):
    jobs = {
        "stage4_lora_probe": job_status(args.probe_job_id),
        "stage5_smoke": job_status(args.smoke_job_id),
        "stage5_multi_prompt": job_status(args.multi_job_id),
        "stage6_transfer": job_status(args.transfer_job_id),
    }
    artifacts = artifact_status(args)
    return {
        "schema_version": "ascr.phase56_monitor.v1",
        "created_at_utc": utc_now(),
        "jobs": jobs,
        "artifacts": artifacts,
        "recommendation": recommendation(jobs, artifacts),
    }


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"status_{stamp}.json"
    latest = output_dir / "latest.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")
    return path


def build_parser():
    parser = argparse.ArgumentParser(description="Monitor ASCR Phase-5/6 jobs and artifacts.")
    parser.add_argument("--probe-job-id", default="161672")
    parser.add_argument("--smoke-job-id", default="161688")
    parser.add_argument("--multi-job-id", default="161689")
    parser.add_argument("--transfer-job-id", default="161690")
    parser.add_argument("--stage5-smoke-dir", default="outputs/stage5_self_corrupt/token_repair_8x8_smoke_epoch3")
    parser.add_argument("--stage5-multi-dir", default="outputs/stage5_self_corrupt/token_repair_8x8_epoch3_multi_prompt")
    parser.add_argument("--stage6-transfer-dir", default="outputs/stage6_transfer/token_repair_8x8_epoch3_limit8")
    parser.add_argument("--output-dir", default="outputs/monitor/phase5_6")
    parser.add_argument("--watch-interval", type=int, default=0)
    parser.add_argument("--max-checks", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checks = 0
    while True:
        checks += 1
        report = build_report(args)
        path = write_report(report, args.output_dir)
        print(json.dumps({
            "report": str(path),
            "recommendation": report["recommendation"],
            "created_at_utc": report["created_at_utc"],
        }, sort_keys=True))
        if args.watch_interval <= 0 or (args.max_checks and checks >= args.max_checks):
            break
        time.sleep(int(args.watch_interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

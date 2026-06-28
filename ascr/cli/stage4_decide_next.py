"""Generate next-action recommendations from Stage-4 server outputs."""

from __future__ import annotations

import argparse
from glob import glob
import json
from pathlib import Path

from ascr.analysis.stage4_run_decision import (
    decide_stage4_next_actions,
    load_json_files,
    read_json,
    scan_log_files,
    write_next_actions,
)


def _expand_patterns(patterns):
    paths = []
    for pattern in patterns or []:
        matched = sorted(glob(pattern, recursive=True))
        if matched:
            paths.extend(matched)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(str(candidate))
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description="Decide the next Stage-4 server actions from registry/failure/log outputs.")
    parser.add_argument("--registry", required=True, help="Path to stage4_run_registry.json.")
    parser.add_argument("--failure-summary", action="append", default=[], help="Path or glob for failure_summary.json files.")
    parser.add_argument("--log", action="append", default=[], help="Path or glob for Slurm/log files to scan.")
    parser.add_argument("--parse-threshold", type=float, default=0.5)
    parser.add_argument("--hit-any-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    registry = read_json(args.registry)
    failure_paths = _expand_patterns(args.failure_summary)
    log_paths = _expand_patterns(args.log)
    decision = decide_stage4_next_actions(
        registry,
        failure_summaries=load_json_files(failure_paths),
        log_scan=scan_log_files(log_paths),
        parse_threshold=args.parse_threshold,
        hit_any_threshold=args.hit_any_threshold,
    )
    outputs = write_next_actions(args.output_dir, decision)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Route Stage-4 failure modes to concrete repair actions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def route_stage4_failure(parse_rate, hit_any_rate, malformed_rate=None):
    parse_rate = float(parse_rate or 0.0)
    hit_any_rate = float(hit_any_rate or 0.0)
    malformed_rate = float(malformed_rate if malformed_rate is not None else max(0.0, 1.0 - parse_rate))
    if parse_rate < 0.3:
        return {
            "route": "prompt_decoding_fix",
            "priority": 10,
            "reason": "parse_rate < 0.3",
            "command": "MODE=diagnostic_sweep bash scripts/training/run_stage4_server_campaign.sh",
        }
    if parse_rate > 0.5 and hit_any_rate < 0.1:
        return {
            "route": "more_data_or_capacity",
            "priority": 20,
            "reason": "format is mostly usable but localization hit_any is weak",
            "command": "PROFILE=l40s_1024_gc MODE=split_curriculum bash scripts/training/run_stage4_server_campaign.sh",
        }
    if parse_rate > 0.7 and hit_any_rate > 0.2:
        return {
            "route": "scale_grid",
            "priority": 30,
            "reason": "parse and localization both clear the scale gate",
            "command": "PROFILE=l40s_1024_gc sbatch --array=1-2 jobs/stage4/train_mmu_lora_curriculum.sbatch",
        }
    if hit_any_rate > 0.5:
        return {
            "route": "phase5_loop",
            "priority": 40,
            "reason": "localizer is strong enough for closed-loop repair",
            "command": "bash scripts/training/run_stage5_loop.sh",
        }
    return {
        "route": "diagnose_failure_rows",
        "priority": 50,
        "reason": "metrics are inconclusive",
        "command": "bash scripts/training/run_stage4_postprocess.sh",
    }


def route_from_summary(summary):
    row_count = int(summary.get("row_count") or 0)
    malformed = int(summary.get("malformed_count") or 0)
    malformed_rate = malformed / row_count if row_count else None
    metrics = summary.get("metrics") or {}
    route = route_stage4_failure(
        summary.get("parse_rate"),
        metrics.get("hit_any_rate", summary.get("hit_any_rate")),
        malformed_rate=malformed_rate,
    )
    return {
        "schema_version": "ascr.stage4.failure_route.v1",
        "created_at_utc": created_at_utc(),
        "summary_path": summary.get("_source_path"),
        "parse_rate": float(summary.get("parse_rate") or 0.0),
        "hit_any_rate": float(metrics.get("hit_any_rate", summary.get("hit_any_rate") or 0.0)),
        "malformed_rate": malformed_rate,
        **route,
    }


def write_route(output_dir, route):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage4_failure_route.json"
    md_path = output_dir / "stage4_failure_route.md"
    json_path.write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# Stage-4 Failure Route",
            "",
            f"- Route: `{route['route']}`",
            f"- Priority: `{route['priority']}`",
            f"- Reason: {route['reason']}",
            "",
            "```bash",
            route["command"],
            "```",
            "",
        ]),
        encoding="utf-8",
    )
    return {"route_json": str(json_path), "route_md": str(md_path)}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Route a Stage-4 probe summary to the next repair action.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    summary["_source_path"] = args.summary
    route = route_from_summary(summary)
    print(json.dumps(write_route(args.output_dir, route), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


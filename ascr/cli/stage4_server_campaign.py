"""Generate server-side Stage-4 campaign plans for ASCR MMU LoRA runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import stat


DEFAULT_OUTPUT_DIR = "outputs/stage4_self_corrupt/campaigns/stage4_h200_1024_schema_example"
DEFAULT_PROFILE = "h200_1024"
DEFAULT_GRIDS = (4, 8, 16)
GRID_TO_ARRAY_INDEX = {4: 0, 8: 1, 16: 2}


def profile_paths(profile):
    if profile in {"h200", "h200_1024", "current_server"}:
        return {
            "campaign": "stage4_h200_1024_schema_example",
            "train_dir": "lora_h200_1024px_adamw",
            "probe_dir": "probe_lora_h200_1024px_eval",
            "title": "Submit schema_example H200 1024px curriculum array",
        }
    if profile == "l40s_1024_gc":
        return {
            "campaign": "stage4_1024gc_schema_example",
            "train_dir": "lora_l40s_1024px_gc_adam8bit",
            "probe_dir": "probe_lora_l40s_1024px_gc_eval",
            "title": "Submit schema_example 1024px GC curriculum array",
        }
    if profile == "l40s":
        return {
            "campaign": "stage4_l40s_schema_example",
            "train_dir": "lora_l40s",
            "probe_dir": "probe_lora_l40s_eval",
            "title": "Submit schema_example L40S fallback curriculum array",
        }
    raise ValueError(f"Unsupported campaign profile: {profile}")


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_grids(value):
    if value is None:
        return list(DEFAULT_GRIDS)
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).replace(",", " ").split()
    grids = [int(part) for part in parts if str(part).strip()]
    unsupported = [grid for grid in grids if grid not in GRID_TO_ARRAY_INDEX]
    if unsupported:
        raise ValueError(f"Unsupported Stage-4 curriculum grids for Slurm array: {unsupported}")
    return grids


def _array_selector(grids):
    indices = sorted(GRID_TO_ARRAY_INDEX[int(grid)] for grid in grids)
    if not indices:
        raise ValueError("At least one grid is required.")
    if indices == list(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}-{indices[-1]}" if len(indices) > 1 else str(indices[0])
    return ",".join(str(index) for index in indices)


def _split_submit_commands(grids, profile):
    return [
        {
            "grid": int(grid),
            "array_index": GRID_TO_ARRAY_INDEX[int(grid)],
            "command": (
                f"PROFILE={profile} "
                f"sbatch --parsable --array={GRID_TO_ARRAY_INDEX[int(grid)]} "
                "jobs/stage4/train_mmu_lora_curriculum.sbatch"
            ),
        }
        for grid in grids
    ]


def build_campaign_plan(
    *,
    grids=None,
    profile=DEFAULT_PROFILE,
    output_dir=DEFAULT_OUTPUT_DIR,
    include_decoding_diagnostic=True,
):
    grids = parse_grids(grids)
    profile_info = profile_paths(profile)
    grid_text = " ".join(str(grid) for grid in grids)
    array_selector = _array_selector(grids)
    primary_submit = (
        f"PROFILE={profile} "
        f"sbatch --parsable --array={array_selector} "
        "jobs/stage4/train_mmu_lora_curriculum.sbatch"
    )
    summarize = (
        f"PROFILE={profile} RUN_ZERO_PROBE=0 RUN_PREP=0 RUN_CONVERT=0 "
        f"RUN_LORA_TRAIN=0 RUN_LORA_PROBE=0 RUN_SUMMARY=1 GRIDS='{grid_text}' "
        "bash scripts/training/run_stage4_curriculum.sh"
    )
    postprocess = "bash scripts/training/run_stage4_postprocess.sh"
    steps = [
        {
            "id": "server_preflight",
            "title": "Verify server checkout and lightweight Python health",
            "intent": "Catch stale checkouts or broken environments before GPU jobs consume queue slots.",
            "commands": [
                "git status --short --branch",
                "python -m compileall ascr",
                "python scripts/smoke_test.py",
            ],
        },
        {
            "id": "submit_schema_example_curriculum",
            "title": profile_info["title"],
            "intent": (
                "Regenerate SFT with the schema_example default, convert to Lumina data, "
                "train LoRA adapters, and probe each requested grid."
            ),
            "commands": [primary_submit],
            "fallback_commands": _split_submit_commands(grids, profile),
            "wait_for_completion": True,
        },
        {
            "id": "summarize_and_decide",
            "title": "Summarize completed grid runs and update Stage-4 decisions",
            "intent": "Build curriculum summaries, registry rows, failure analyses, and next-action files.",
            "commands": [summarize, postprocess],
        },
    ]
    if include_decoding_diagnostic:
        steps.append(
            {
                "id": "optional_schema_example_decoding_diagnostic",
                "title": "Run schema_example decoding diagnostic only if JSON formatting remains weak",
                "intent": (
                    "Keep the old sweep as a narrow diagnostic over answer length, not as "
                    "the main prompt-selection strategy."
                ),
                "commands": [
                    "MODE=plan bash scripts/training/run_stage4_probe_sweep.sh",
                    "sbatch jobs/stage4/stage4_probe_sweep.sbatch",
                    "MODE=summarize bash scripts/training/run_stage4_probe_sweep.sh",
                ],
                "condition": "Use only when postprocess still reports malformed JSON under schema_example.",
            }
        )
    return {
        "schema_version": "ascr.stage4.server_campaign.v1",
        "created_at_utc": created_at_utc(),
        "campaign": profile_info["campaign"],
        "output_dir": str(output_dir),
        "prompt_policy": "schema_example is the Stage-4 default; legacy variants are reproducibility-only.",
        "profile": profile,
        "grids": grids,
        "slurm_array": array_selector,
        "primary_submit_command": primary_submit,
        "split_submit_commands": _split_submit_commands(grids, profile),
        "steps": steps,
        "expected_outputs": [
            f"outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid{grid}/vq_tokens/{profile_info['train_dir']}/training_manifest.json"
            for grid in grids
        ]
        + [
            f"outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid{grid}/vq_tokens/{profile_info['probe_dir']}/summary.json"
            for grid in grids
        ],
    }


def _markdown(plan):
    lines = [
        "# Stage-4 Server Campaign",
        "",
        f"- Campaign: `{plan['campaign']}`",
        f"- Profile: `{plan['profile']}`",
        f"- Grids: `{', '.join(str(grid) for grid in plan['grids'])}`",
        f"- Slurm array selector: `{plan['slurm_array']}`",
        f"- Prompt policy: {plan['prompt_policy']}",
        "",
        "## Primary Submit",
        "",
        "```bash",
        plan["primary_submit_command"],
        "```",
        "",
        "## QOS Split Submit Fallback",
        "",
        "```bash",
    ]
    lines.extend(item["command"] for item in plan["split_submit_commands"])
    lines.extend(["```", "", "## Steps", ""])
    for index, step in enumerate(plan["steps"], start=1):
        lines.extend([
            f"### {index}. {step['title']}",
            "",
            step["intent"],
            "",
            "```bash",
        ])
        lines.extend(step["commands"])
        lines.extend(["```", ""])
        if step.get("condition"):
            lines.extend([f"Condition: {step['condition']}", ""])
    lines.extend(["## Expected Outputs", ""])
    lines.extend(f"- `{path}`" for path in plan["expected_outputs"])
    lines.append("")
    return "\n".join(lines)


def _shell_script(plan):
    split_commands = "\n".join(item["command"] for item in plan["split_submit_commands"])
    return f"""#!/usr/bin/env bash
# Server helper for {plan['campaign']}.

set -euo pipefail

PROJECT_ROOT=${{PROJECT_ROOT:-$(git rev-parse --show-toplevel)}}
cd "$PROJECT_ROOT"
mkdir -p logs

export PROFILE=${{PROFILE:-{plan['profile']}}}
export LUMINA_REPO=${{LUMINA_REPO:-third_party/Lumina-DiMOO}}
export LUMINA_MODEL_PATH=${{LUMINA_MODEL_PATH:-models/lumina-dimoo}}

MODE=${{MODE:-plan}}  # plan, submit_curriculum, split_curriculum, summarize, diagnostic_sweep

case "$MODE" in
  plan)
    cat <<'CMDS'
{plan['primary_submit_command']}

# If QOS rejects the combined array, submit split jobs:
{split_commands}

# After all curriculum jobs finish:
PROFILE={plan['profile']} RUN_ZERO_PROBE=0 RUN_PREP=0 RUN_CONVERT=0 RUN_LORA_TRAIN=0 RUN_LORA_PROBE=0 RUN_SUMMARY=1 GRIDS='{" ".join(str(grid) for grid in plan['grids'])}' bash scripts/training/run_stage4_curriculum.sh
bash scripts/training/run_stage4_postprocess.sh
CMDS
    ;;
  submit_curriculum)
    {plan['primary_submit_command']}
    ;;
  split_curriculum)
{chr(10).join("    " + item["command"] for item in plan["split_submit_commands"])}
    ;;
  summarize)
    PROFILE={plan['profile']} RUN_ZERO_PROBE=0 RUN_PREP=0 RUN_CONVERT=0 RUN_LORA_TRAIN=0 RUN_LORA_PROBE=0 RUN_SUMMARY=1 GRIDS='{" ".join(str(grid) for grid in plan['grids'])}' bash scripts/training/run_stage4_curriculum.sh
    bash scripts/training/run_stage4_postprocess.sh
    ;;
  diagnostic_sweep)
    MODE=plan bash scripts/training/run_stage4_probe_sweep.sh
    sbatch jobs/stage4/stage4_probe_sweep.sbatch
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac
"""


def write_campaign_outputs(output_dir, plan):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "campaign_manifest.json"
    md_path = output_dir / "campaign_plan.md"
    shell_path = output_dir / "run_stage4_server_campaign.sh"
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(plan), encoding="utf-8")
    shell_path.write_text(_shell_script(plan), encoding="utf-8", newline="\n")
    shell_path.chmod(shell_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {
        "campaign_manifest": str(json_path),
        "campaign_plan": str(md_path),
        "campaign_shell": str(shell_path),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Stage-4 server campaign manifests and shell commands.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--grids", default=",".join(str(grid) for grid in DEFAULT_GRIDS))
    parser.add_argument("--include-decoding-diagnostic", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = build_campaign_plan(
        grids=parse_grids(args.grids),
        profile=args.profile,
        output_dir=args.output_dir,
        include_decoding_diagnostic=args.include_decoding_diagnostic,
    )
    outputs = write_campaign_outputs(args.output_dir, plan)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

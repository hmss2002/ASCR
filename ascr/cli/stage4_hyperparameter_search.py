"""Generate Stage-4 MMU LoRA hyperparameter search trials."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import re

from ascr.core.config import load_config


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _dump_simple_yaml(path, payload):
    lines = []
    for key, value in payload.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif value is None:
            value = "null"
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_trials(base_config, search_space, trials, output_dir, seed=0, mode="random"):
    keys = list(search_space)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(search_space[key] for key in keys))]
    rng = random.Random(seed)
    if mode == "random":
        rng.shuffle(combos)
    combos = combos[: int(trials)]
    output_dir = Path(output_dir)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, overrides in enumerate(combos):
        config = dict(base_config)
        config.update(overrides)
        config["epochs"] = int(overrides.get("epochs", min(int(config.get("epochs", 1)), 2)))
        config["output_dir"] = str(output_dir / "trials" / f"trial_{index:03d}")
        config_path = config_dir / f"trial_{index:03d}.yaml"
        _dump_simple_yaml(config_path, config)
        rows.append({
            "trial_index": index,
            "config": str(config_path),
            "overrides": overrides,
            "output_dir": config["output_dir"],
            "command": f"python -m ascr.cli.stage4_train_mmu_lora --config {config_path}",
        })
    plan = {
        "schema_version": "ascr.stage4.hyperparameter_search.v1",
        "created_at_utc": created_at_utc(),
        "trial_count": len(rows),
        "mode": mode,
        "seed": int(seed),
        "trials": rows,
    }
    return plan


def write_outputs(output_dir, plan):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "hparam_search_plan.json"
    shell_path = output_dir / "run_hparam_trials.sh"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shell_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(row["command"] for row in plan["trials"])
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"plan": str(plan_path), "shell": str(shell_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Stage-4 hyperparameter search trial configs.")
    parser.add_argument("--search-space", required=True, help="JSON object of key -> list values.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["random", "grid"], default="random")
    return parser


def parse_search_space(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except Exception:
            quoted = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', value)
            return ast.literal_eval(quoted)


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = build_trials(
        load_config(args.config),
        parse_search_space(args.search_space),
        args.trials,
        args.output_dir,
        seed=args.seed,
        mode=args.mode,
    )
    print(json.dumps(write_outputs(args.output_dir, plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

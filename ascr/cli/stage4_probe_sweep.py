"""Run or summarize Stage-4 prompt/decoding sweeps for an existing LoRA adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from ascr.core.config import load_config
from ascr.training.stage4_mmu_lora import PROMPT_VARIANT_DEFAULT, normalise_prompt_variant, run_mmu_localization_probe


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_csv(value, cast=str):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def _combo_label(combo):
    return (
        f"{combo['prompt_variant']}"
        f"_tok{combo['max_new_tokens']}"
        f"_steps{combo['answer_steps']}"
        f"_temp{str(combo['answer_temperature']).replace('.', 'p')}"
        f"_cfg{str(combo['answer_cfg_scale']).replace('.', 'p')}"
    )


def build_sweep_plan(
    base_config,
    output_root,
    prompt_variants=None,
    max_new_tokens=None,
    answer_steps=None,
    answer_temperatures=None,
    answer_cfg_scales=None,
    answer_block_lengths=None,
):
    prompt_variants = [normalise_prompt_variant(item) for item in (prompt_variants or [PROMPT_VARIANT_DEFAULT])]
    max_new_tokens = [int(item) for item in (max_new_tokens or [base_config.get("max_new_tokens", 384)])]
    answer_steps = [int(item) for item in (answer_steps or [base_config.get("answer_steps", 64)])]
    answer_temperatures = [float(item) for item in (answer_temperatures or [base_config.get("answer_temperature", 0.0)])]
    answer_cfg_scales = [float(item) for item in (answer_cfg_scales or [base_config.get("answer_cfg_scale", 0.0)])]
    answer_block_lengths = [int(item) for item in (answer_block_lengths or [base_config.get("answer_block_length", 128)])]
    combos = []
    for prompt_variant in prompt_variants:
        for tokens in max_new_tokens:
            for steps in answer_steps:
                for temp in answer_temperatures:
                    for cfg in answer_cfg_scales:
                        for block_len in answer_block_lengths:
                            combo = {
                                "index": len(combos),
                                "prompt_variant": prompt_variant,
                                "max_new_tokens": int(tokens),
                                "answer_steps": int(steps),
                                "answer_temperature": float(temp),
                                "answer_cfg_scale": float(cfg),
                                "answer_block_length": int(block_len),
                            }
                            combo["label"] = _combo_label(combo)
                            combo["output_dir"] = str(Path(output_root) / combo["label"])
                            combos.append(combo)
    return {
        "schema_version": "ascr.stage4.probe_sweep.plan.v1",
        "created_at_utc": _created_at(),
        "base_config": base_config,
        "output_root": str(output_root),
        "combo_count": len(combos),
        "combos": combos,
    }


def _run_combo(base_config, combo):
    config = dict(base_config)
    config.update({
        "output_dir": combo["output_dir"],
        "prompt_variant": combo["prompt_variant"],
        "max_new_tokens": combo["max_new_tokens"],
        "answer_steps": combo["answer_steps"],
        "answer_temperature": combo["answer_temperature"],
        "answer_cfg_scale": combo["answer_cfg_scale"],
        "answer_block_length": combo["answer_block_length"],
    })
    return run_mmu_localization_probe(
        config.get("dataset"),
        config.get("output_dir"),
        grid_size=int(config.get("grid_size", 16)),
        max_selected_cells=int(config.get("max_selected_cells", 16)),
        top_k=int(config.get("top_k", 4)),
        limit=config.get("limit"),
        sample_ids=config.get("sample_ids"),
        split_manifest=config.get("split_manifest"),
        split=config.get("split", "eval"),
        input_mode=config.get("input_mode"),
        use_vq_tokens=bool(config.get("use_vq_tokens", True)),
        target_schema=config.get("target_schema", "localization_cells"),
        prompt_variant=config.get("prompt_variant", PROMPT_VARIANT_DEFAULT),
        lora_path=config.get("lora_path"),
        repo_path=config.get("repo_path", os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO")),
        checkpoint_path=config.get("checkpoint_path", os.environ.get("LUMINA_MODEL_PATH", "models/lumina-dimoo")),
        device=config.get("device", "cuda"),
        image_size=int(config.get("image_size", 1024)),
        max_new_tokens=int(config.get("max_new_tokens", 384)),
        answer_steps=int(config.get("answer_steps", 64)),
        answer_block_length=int(config.get("answer_block_length", 128)),
        answer_temperature=float(config.get("answer_temperature", 0.0)),
        answer_cfg_scale=float(config.get("answer_cfg_scale", 0.0)),
    )


def summarize_sweep(plan):
    rows = []
    for combo in plan.get("combos", []):
        summary_path = Path(combo["output_dir"]) / "summary.json"
        row = dict(combo)
        row["summary_path"] = str(summary_path)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary.get("metrics") or {}
            row.update({
                "status": "complete",
                "row_count": summary.get("row_count"),
                "parse_rate": summary.get("parse_rate"),
                "malformed_count": summary.get("malformed_count"),
                "call_error_count": summary.get("call_error_count"),
                "hit_any_rate": metrics.get("hit_any_rate"),
                "mean_f1_at_k": metrics.get("mean_f1_at_k"),
                "mean_iou": metrics.get("mean_iou"),
            })
        else:
            row["status"] = "missing"
        rows.append(row)
    complete_rows = [row for row in rows if row.get("status") == "complete"]
    best = sorted(
        complete_rows,
        key=lambda row: (
            float(row.get("parse_rate") or 0.0),
            float(row.get("hit_any_rate") or 0.0),
            -float(row.get("malformed_count") or 0.0),
        ),
        reverse=True,
    )[:5]
    return {
        "schema_version": "ascr.stage4.probe_sweep.summary.v1",
        "created_at_utc": _created_at(),
        "combo_count": len(rows),
        "complete_count": len(complete_rows),
        "rows": rows,
        "best": best,
    }


def write_plan(output_dir, plan):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "probe_sweep_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(plan_path)


def write_summary(output_dir, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "probe_sweep_summary.json"
    md_path = output_dir / "probe_sweep_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-4 Probe Sweep Summary",
        "",
        "| Label | Status | Parse | Hit any | F1 | IoU | Malformed | Calls |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {label} | {status} | {parse} | {hit} | {f1} | {iou} | {malformed} | {calls} |".format(
                label=row["label"],
                status=row["status"],
                parse=row.get("parse_rate", ""),
                hit=row.get("hit_any_rate", ""),
                f1=row.get("mean_f1_at_k", ""),
                iou=row.get("mean_iou", ""),
                malformed=row.get("malformed_count", ""),
                calls=row.get("call_error_count", ""),
            )
        )
    lines.extend(["", "## Best Complete Runs", ""])
    for row in summary.get("best", []):
        lines.append(f"- `{row['label']}` parse={row.get('parse_rate')} hit_any={row.get('hit_any_rate')} summary=`{row.get('summary_path')}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"sweep_summary_json": str(json_path), "sweep_summary_md": str(md_path)}


def _load_or_build_plan(args):
    if args.plan:
        return json.loads(Path(args.plan).read_text(encoding="utf-8"))
    base_config = load_config(args.config)
    return build_sweep_plan(
        base_config,
        args.output_root,
        prompt_variants=_parse_csv(args.prompt_variants, str),
        max_new_tokens=_parse_csv(args.max_new_tokens, int),
        answer_steps=_parse_csv(args.answer_steps, int),
        answer_temperatures=_parse_csv(args.answer_temperatures, float),
        answer_cfg_scales=_parse_csv(args.answer_cfg_scales, float),
        answer_block_lengths=_parse_csv(args.answer_block_lengths, int),
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run or summarize Stage-4 prompt/decoding probe sweeps.")
    parser.add_argument("--config", default="configs/stage4/self_corrupt/mmu_probe_lora_hard64_grid4_vq_tokens_l40s_1024px_gc.yaml")
    parser.add_argument("--output-root", default="outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_sweep_l40s_1024px_gc")
    parser.add_argument("--prompt-variants", default=PROMPT_VARIANT_DEFAULT)
    parser.add_argument("--max-new-tokens", default="384,512")
    parser.add_argument("--answer-steps", default="64")
    parser.add_argument("--answer-temperatures", default="0.0")
    parser.add_argument("--answer-cfg-scales", default="0.0")
    parser.add_argument("--answer-block-lengths", default="128")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--write-plan-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--combo-index", type=int, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = _load_or_build_plan(args)
    plan_path = write_plan(args.output_root, plan)
    if args.write_plan_only:
        print(json.dumps({"plan": plan_path, "combo_count": plan["combo_count"]}, indent=2, sort_keys=True))
        return 0
    if args.summarize_only:
        outputs = write_summary(args.output_root, summarize_sweep(plan))
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0
    combo_index = args.combo_index
    if combo_index is None:
        combo_index = int(os.environ.get("SWEEP_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    combo = plan["combos"][int(combo_index)]
    summary = _run_combo(plan["base_config"], combo)
    sweep_summary = write_summary(args.output_root, summarize_sweep(plan))
    print(json.dumps({"plan": plan_path, "combo": combo, "summary": summary, **sweep_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

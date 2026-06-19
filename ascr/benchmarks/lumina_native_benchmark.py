import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from ascr.core.config import load_config
from ascr.core.loop import ASCRLoop, run_config_from_mapping
from ascr.evaluators.lumina_native import attach_lumina_native_engine_if_available
from ascr.evaluators.registry import build_evaluator
from ascr.generators.registry import build_generator
from ascr.revision.selector import GridSemanticReopeningSelector


def read_prompts(path, limit=None):
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            prompts.append(text)
        if limit is not None and len(prompts) >= int(limit):
            break
    return prompts


def shard_rows(rows, shard_index=0, shard_count=1):
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    return [row for index, row in enumerate(rows) if index % shard_count == shard_index]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def selected_token_counts(summary):
    return [record.get("selected_token_count", 0) for record in summary.get("revision_records", [])]


def build_loop(config, generator_name="lumina"):
    default_token_grid_size = {"mock": 16, "lumina": 64}.get(generator_name, 64)
    default_image_size = {"mock": 256, "lumina": 1024}.get(generator_name, 1024)
    generator_config = dict(config)
    generator_config["token_grid_size"] = int(config.get("token_grid_size", default_token_grid_size))
    generator_config["image_size"] = int(config.get("image_size", default_image_size))
    config["token_grid_size"] = generator_config["token_grid_size"]
    config["image_size"] = generator_config["image_size"]
    config.setdefault("evaluator", {})
    config["evaluator"]["name"] = "lumina_native_evaluator"
    generator = build_generator(generator_name, generator_config)
    evaluator = build_evaluator("lumina_native_evaluator", config)
    attach_lumina_native_engine_if_available(generator, evaluator)
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get("coarse_grid_size", 4)),
        token_grid_size=int(config.get("token_grid_size", 64)),
        dilation=int(config.get("dilation", 1)),
    )
    return ASCRLoop(generator, evaluator, selector, run_config_from_mapping(config))


def run_benchmark(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.config) if args.config else {}
    base_config.setdefault("generator", {})
    base_config.setdefault("evaluator", {})
    base_config["evaluator"]["name"] = "lumina_native_evaluator"
    if args.output_dir:
        base_config["output_dir"] = str(output_dir / "runs")
    if args.max_iterations is not None:
        base_config["max_iterations"] = int(args.max_iterations)
    if args.run_name:
        base_config["run_name"] = args.run_name
    generator_name = args.generator or base_config.get("generator", {}).get("name", "lumina")
    prompts = shard_rows(
        read_prompts(args.prompts, limit=args.limit),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    rows = []
    errors = []
    for local_index, prompt in enumerate(prompts):
        sample_id = f"{args.domain}:s{int(args.shard_index):02d}:p{local_index:04d}"
        try:
            per_prompt_config = deepcopy(base_config)
            per_prompt_config["output_dir"] = str(output_dir / "runs" / sample_id.replace(":", "_"))
            loop = build_loop(per_prompt_config, generator_name=generator_name)
            summary = loop.run(prompt, project_root=Path.cwd())
            after_image = summary.get("raw_final_decoded_image") or summary.get("final_decoded_image")
            after_grid_image = summary.get("raw_final_grid_image") or summary.get("final_grid_image")
            rows.append({
                "schema_version": "ascr.lumina_native_image_benchmark.v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "domain": args.domain,
                "sample_id": sample_id,
                "prompt": prompt,
                "generator": generator_name,
                "evaluator_backend": "lumina_native_evaluator",
                "before_image": summary.get("initial_decoded_image"),
                "before_grid_image": summary.get("initial_grid_image"),
                "after_image": after_image,
                "after_grid_image": after_grid_image,
                "selected_after_image": summary.get("final_decoded_image"),
                "selected_after_grid_image": summary.get("final_grid_image"),
                "fallback_applied": bool(summary.get("fallback_applied", False)),
                "final_selection_policy": summary.get("final_selection_policy"),
                "stop_reason": summary.get("stop_reason"),
                "evaluator_calls": summary.get("evaluator_calls"),
                "iterations_recorded": summary.get("iterations_recorded"),
                "selected_token_counts": selected_token_counts(summary),
                "artifact_root": summary.get("artifact_root"),
                "trace_path": summary.get("trace_path"),
            })
        except Exception as exc:
            errors.append({
                "domain": args.domain,
                "sample_id": sample_id,
                "prompt": prompt,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            })
            if not args.keep_going:
                raise
    write_jsonl(output_dir / "manifest.jsonl", rows)
    write_jsonl(output_dir / "errors.jsonl", errors)
    summary = {
        "schema_version": "ascr.lumina_native_image_benchmark.summary.v1",
        "domain": args.domain,
        "prompts": str(args.prompts),
        "generator": generator_name,
        "evaluator_backend": "lumina_native_evaluator",
        "row_count": len(rows),
        "error_count": len(errors),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Run before/after ASCR image benchmarks with Lumina-native semantic evaluation.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/stage2/lumina/lumina_native_evaluator_smoke.yaml")
    parser.add_argument("--generator", default=None, choices=["mock", "lumina"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--run-name", default="lumina_native_image_benchmark")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main(argv=None):
    summary = run_benchmark(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

from ascr.benchmarks.metrics import compare_scores, score_image
from ascr.cli.compare_showo_ascr import (
    apply_cli_overrides,
    build_baseline_generator,
    build_loop,
    build_loop_components,
    build_native_baseline,
    load_prompts,
    prompt_run_dir,
    release_cuda_cache,
    resolve_ascr_start_mode,
    share_generator_engine,
    trace_record_count,
)
from ascr.core.config import load_config
from ascr.core.loop import run_config_from_mapping
from ascr.core.loop_direct import DirectTokenReopenLoop
from ascr.evaluators.registry import build_evaluator
from ascr.generators.registry import build_generator
from ascr.revision.selector import DirectTokenReopeningSelector


def build_parser():
    parser = argparse.ArgumentParser(description="Compare Show-o baseline vs coarse-grid ASCR vs direct-token ASCR Stage 1.")
    parser.add_argument("--config", default="configs/stage1/showo/stage1_showo_qwen35_9b_direct_token.yaml")
    parser.add_argument("--coarse-config", default="configs/stage1/showo/stage1_showo_qwen35_9b_fullcap_parallel.yaml", help="Config for the existing coarse-grid ASCR arm; set to 'none' to skip that arm.")
    parser.add_argument("--prompt", default="A red cube left of a blue sphere")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/benchmarks_direct")
    parser.add_argument("--generation-timesteps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--ascr-start-mode", choices=["baseline", "partial"], default=None)
    parser.add_argument("--return-initial-on-max-error", action="store_true")
    parser.add_argument("--reuse-models", action="store_true")
    parser.add_argument("--arms", default="coarse,direct", help="Comma-separated subset of {coarse,direct} to run. Selecting only one arm loads only that arm's Qwen (one model per card). baseline is always generated.")
    return parser


def parse_arms(raw):
    arms = [a.strip().lower() for a in str(raw).split(",") if a.strip()]
    invalid = [a for a in arms if a not in ("coarse", "direct")]
    if invalid:
        raise ValueError(f"Unknown arm(s) {invalid}; valid arms are coarse, direct")
    if not arms:
        raise ValueError("At least one arm must be selected")
    ordered = [a for a in ("direct", "coarse") if a in arms]
    return ordered


def build_direct_loop(config, generator=None, evaluator=None):
    if generator is None or evaluator is None:
        token_grid_size = int(config.get("token_grid_size", 32))
        generator_config = dict(config)
        generator_config["token_grid_size"] = token_grid_size
        generator_config["image_size"] = int(config.get("image_size", 512))
        built_generator = build_generator(config.get("generator", {}).get("name", "showo"), generator_config)
        built_evaluator = build_evaluator(config.get("evaluator", {}).get("name", "qwen_vl_token"), config)
        generator = generator or built_generator
        evaluator = evaluator or built_evaluator
    token_grid_size = int(config.get("token_grid_size", 32))
    selector = DirectTokenReopeningSelector(
        token_grid_size=token_grid_size,
        select_grid_size=int(config.get("select_grid_size", config.get("selector", {}).get("select_grid_size", token_grid_size))),
        dilation=int(config.get("dilation", config.get("selector", {}).get("dilation", 0))),
    )
    return DirectTokenReopenLoop(generator, evaluator, selector, run_config_from_mapping(config), label_step=int(config.get("label_step", 4)))


def _arm_result(prompt, label, summary, image_path, baseline_score, grid_size, image_size):
    score = score_image(prompt, image_path, grid_size=grid_size, image_size=image_size)
    return {
        "arm": label,
        "final_image": image_path,
        "summary": summary,
        "score": score,
        "comparison_vs_baseline": compare_scores(baseline_score, score),
        "evaluator_calls": int(summary.get("evaluator_calls", trace_record_count(summary.get("trace_path", "")))),
        "insertions": int(summary.get("iterations_recorded", 0)),
        "stop_reason": summary.get("stop_reason"),
    }


def run_prompt_three_way(direct_config, coarse_config, prompt, root, args, shared=None, selected_arms=None):
    shared = shared or {}
    selected_arms = selected_arms or ["direct", "coarse"]
    config = apply_cli_overrides(direct_config, args)
    start_mode = resolve_ascr_start_mode(config, args.ascr_start_mode)
    baseline_state, baseline_path = build_native_baseline(config, prompt, root, baseline_generator=shared.get("baseline_generator"))
    image_size = int(config.get("image_size", 512))
    coarse_grid_size = int(config.get("coarse_grid_size", 4))
    baseline_score = score_image(prompt, baseline_path, grid_size=coarse_grid_size, image_size=image_size)

    arms = []

    if "direct" in selected_arms:
        direct_run_config = copy.deepcopy(config)
        direct_run_config["output_dir"] = str(root / "ascr_direct")
        direct_run_config["run_name"] = "stage1_direct_token"
        initial_state = baseline_state if start_mode == "baseline" else None
        direct_summary = build_direct_loop(direct_run_config, generator=shared.get("direct_generator"), evaluator=shared.get("direct_evaluator")).run(prompt, project_root=Path.cwd(), initial_state=initial_state)
        arms.append(_arm_result(prompt, "direct_token", direct_summary, direct_summary["final_decoded_image"], baseline_score, coarse_grid_size, image_size))

    if "coarse" in selected_arms and coarse_config is not None:
        coarse_run_config = apply_cli_overrides(coarse_config, args)
        coarse_run_config["output_dir"] = str(root / "ascr_coarse")
        coarse_run_config["run_name"] = "stage1_coarse_grid"
        coarse_initial = baseline_state if resolve_ascr_start_mode(coarse_run_config, args.ascr_start_mode) == "baseline" else None
        coarse_summary = build_loop(coarse_run_config, generator=shared.get("coarse_generator"), evaluator=shared.get("coarse_evaluator")).run(prompt, project_root=Path.cwd(), initial_state=coarse_initial)
        arms.append(_arm_result(prompt, "coarse_grid", coarse_summary, coarse_summary["final_decoded_image"], baseline_score, coarse_grid_size, image_size))

    result = {
        "prompt": prompt,
        "ascr_start_mode": start_mode,
        "baseline_image": str(baseline_path),
        "baseline_score": baseline_score,
        "arms": arms,
        "primary_metric": "qwen_clean_final_pairwise_judge",
        "primary_metric_status": "pending_external_judge",
    }
    result_path = root / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    return result, result_path


def suite_to_markdown(results):
    lines = [
        "| Index | Prompt | Arm | Evaluator calls | Insertions | Stop reason | Baseline | ASCR | Delta | Verdict |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for index, result in enumerate(results):
        prompt_text = result["prompt"].replace("|", "\\|")
        for arm in result["arms"]:
            comparison = arm["comparison_vs_baseline"]
            lines.append(
                f"| {index} | {prompt_text} | {arm['arm']} | {arm['evaluator_calls']} | {arm['insertions']} | "
                f"{arm['stop_reason']} | {comparison['baseline_score']:.3f} | {comparison['ascr_score']:.3f} | "
                f"{comparison['delta']:.3f} | {comparison['verdict']} |"
            )
    return chr(10).join(lines) + chr(10)


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected_arms = parse_arms(args.arms)
    direct_config = load_config(args.config)
    coarse_config = None if str(args.coarse_config).lower() == "none" else load_config(args.coarse_config)
    if "coarse" not in selected_arms:
        coarse_config = None
    prompts = load_prompts(args.prompt, args.prompts_file, args.prompt_limit)
    root = Path(args.output_dir) / datetime.utcnow().strftime("stage1_variants-%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)

    shared = {}
    if args.reuse_models:
        shared_config = apply_cli_overrides(direct_config, args)
        baseline_generator = build_baseline_generator(shared_config)
        shared.update({"baseline_generator": baseline_generator})
        if "direct" in selected_arms:
            token_grid_size = int(shared_config.get("token_grid_size", 32))
            generator_config = dict(shared_config)
            generator_config["token_grid_size"] = token_grid_size
            generator_config["image_size"] = int(shared_config.get("image_size", 512))
            direct_generator = build_generator(shared_config.get("generator", {}).get("name", "showo"), generator_config)
            direct_evaluator = build_evaluator(shared_config.get("evaluator", {}).get("name", "qwen_vl_token"), shared_config)
            share_generator_engine(baseline_generator, direct_generator)
            shared.update({"direct_generator": direct_generator, "direct_evaluator": direct_evaluator})
        if coarse_config is not None:
            coarse_generator, coarse_evaluator = build_loop_components(apply_cli_overrides(coarse_config, args))
            share_generator_engine(baseline_generator, coarse_generator)
            shared.update({"coarse_generator": coarse_generator, "coarse_evaluator": coarse_evaluator})

    results = []
    for index, prompt in enumerate(prompts):
        prompt_root = prompt_run_dir(root, prompt, index, len(prompts))
        prompt_root.mkdir(parents=True, exist_ok=True)
        result, result_path = run_prompt_three_way(direct_config, coarse_config, prompt, prompt_root, args, shared=shared, selected_arms=selected_arms)
        result["result_path"] = str(result_path)
        results.append(result)
        if not args.reuse_models:
            release_cuda_cache()

    suite_path = root / "suite.json"
    suite_path.write_text(json.dumps({"prompt_count": len(results), "results": results}, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    suite_md_path = root / "suite.md"
    suite_md_path.write_text(suite_to_markdown(results), encoding="utf-8")
    print(json.dumps({"suite_path": str(suite_path), "suite_markdown_path": str(suite_md_path), "prompt_count": len(results)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

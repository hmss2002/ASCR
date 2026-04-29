import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path

from ascr.benchmarks.metrics import compare_scores, score_image
from ascr.benchmarks.runner import result_to_markdown
from ascr.core.artifacts import RunArtifacts
from ascr.core.config import load_config
from ascr.core.loop import ASCRLoop, run_config_from_mapping
from ascr.evaluators.registry import build_evaluator
from ascr.generators.registry import build_generator
from ascr.generators.showo import ShowOAdapter
from ascr.revision.selector import GridSemanticReopeningSelector


def build_parser():
    parser = argparse.ArgumentParser(description="Compare native original Show-o against ASCR Stage 1.")
    parser.add_argument("--config", default="configs/stage1_showo_local.yaml")
    parser.add_argument("--prompt", default="A red cube left of a blue sphere")
    parser.add_argument("--prompts-file", default=None, help="Optional newline-delimited prompt suite. Blank lines and # comments are ignored.")
    parser.add_argument("--prompt-limit", type=int, default=None, help="Optional cap for prompt-suite smoke runs.")
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    parser.add_argument("--generation-timesteps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--ascr-start-mode", choices=["baseline", "partial"], default=None, help="baseline starts ASCR from a completed native Show-o sample; partial starts from the configured confidence block so evaluator feedback is inserted during denoising.")
    return parser


ASCR_START_MODES = {"baseline", "partial"}


def load_prompts(prompt, prompts_file=None, limit=None):
    if prompts_file:
        prompts = []
        for raw_line in Path(prompts_file).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    else:
        prompts = [prompt]
    if limit is not None:
        prompts = prompts[:max(0, int(limit))]
    if not prompts:
        raise ValueError("No prompts were provided for comparison.")
    return prompts


def resolve_ascr_start_mode(config, override=None):
    mode = (override or config.get("ascr_start_mode") or "baseline").strip().lower()
    if mode not in ASCR_START_MODES:
        raise ValueError(f"Unknown ASCR start mode {mode!r}; expected one of {sorted(ASCR_START_MODES)}")
    return mode


def prompt_run_dir(root, prompt, index, total):
    if total == 1:
        return root
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48]
    suffix = f"-{slug}" if slug else ""
    return root / f"prompt_{index:03d}{suffix}"


def apply_cli_overrides(config, args):
    config = copy.deepcopy(config)
    config["max_iterations"] = args.max_iterations
    generator_config = config.setdefault("generator", {})
    if args.generation_timesteps is not None:
        generator_config["generation_timesteps"] = args.generation_timesteps
    if args.guidance_scale is not None:
        generator_config["guidance_scale"] = args.guidance_scale
    return config


def trace_record_count(trace_path):
    try:
        return sum(1 for line in Path(trace_path).read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def build_loop(config):
    generator_config = dict(config)
    generator_config["token_grid_size"] = int(config.get("token_grid_size", 32))
    generator_config["image_size"] = int(config.get("image_size", 512))
    generator = build_generator(config.get("generator", {}).get("name", "showo"), generator_config)
    evaluator = build_evaluator(config.get("evaluator", {}).get("name", "local_vlm"), config)
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get("coarse_grid_size", 4)),
        token_grid_size=int(config.get("token_grid_size", 32)),
        dilation=int(config.get("dilation", 1)),
    )
    return ASCRLoop(generator, evaluator, selector, run_config_from_mapping(config))


def build_native_baseline(config, prompt, root):
    generator_config = config.get("generator", {})
    baseline_generator = ShowOAdapter(
        repo_path=generator_config.get("repo_path"),
        checkpoint_path=generator_config.get("checkpoint_path"),
        vq_model_path=generator_config.get("vq_model_path"),
        llm_model_path=generator_config.get("llm_model_path"),
        showo_config_path=generator_config.get("showo_config_path"),
        device=generator_config.get("device", "cuda"),
        token_grid_size=int(config.get("token_grid_size", 32)),
        image_size=int(config.get("image_size", 512)),
        guidance_scale=float(generator_config.get("guidance_scale", 4.0)),
        generation_timesteps=int(generator_config.get("generation_timesteps", 18)),
        seed=int(config.get("seed", generator_config.get("seed", 1234))),
        native_token_loop=True,
        confidence_steps=int(generator_config.get("generation_timesteps", 18)),
    )
    artifacts = RunArtifacts(root / "baseline_native_state")
    baseline_state = baseline_generator.initialize(prompt, artifacts)
    baseline_path = root / "baseline_showo.png"
    baseline_state = baseline_generator.decode(baseline_state, baseline_path)
    baseline_state.metadata["source"] = "native_showo_baseline_compare"
    return baseline_state, baseline_path


def run_prompt_comparison(base_config, prompt, root, args):
    config = apply_cli_overrides(base_config, args)
    start_mode = resolve_ascr_start_mode(config, args.ascr_start_mode)
    baseline_state, baseline_path = build_native_baseline(config, prompt, root)
    config["output_dir"] = str(root / "ascr")
    config["run_name"] = "stage1_showo_ascr"
    initial_state = baseline_state if start_mode == "baseline" else None
    summary = build_loop(config).run(prompt, project_root=Path.cwd(), initial_state=initial_state)
    grid_size = int(config.get("coarse_grid_size", 4))
    image_size = int(config.get("image_size", 512))
    baseline_score = score_image(prompt, baseline_path, grid_size=grid_size, image_size=image_size)
    ascr_score = score_image(prompt, summary["final_decoded_image"], grid_size=grid_size, image_size=image_size)
    comparison = compare_scores(baseline_score, ascr_score)
    result = {
        "prompt": prompt,
        "ascr_start_mode": start_mode,
        "baseline_image": str(baseline_path),
        "ascr_final_image": summary["final_decoded_image"],
        "ascr_summary": summary,
        "evaluator_calls": int(summary.get("evaluator_calls", trace_record_count(summary.get("trace_path", "")))),
        "ascr_insertions": int(summary.get("iterations_recorded", 0)),
        "baseline_score": baseline_score,
        "ascr_score": ascr_score,
        "comparison": comparison,
        "native_baseline_metadata": {
            "confidence_steps": baseline_state.metadata.get("confidence_steps"),
            "confidence_remask_count": baseline_state.metadata.get("confidence_remask_count"),
            "token_state_path": baseline_state.metadata.get("token_state_path"),
            "confidence_path": baseline_state.metadata.get("confidence_path"),
        },
    }
    result_path = root / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    markdown_path = root / "comparison.md"
    markdown_path.write_text(result_to_markdown(result), encoding="utf-8")
    return result, result_path, markdown_path


def build_suite(results):
    verdicts = {}
    for result in results:
        verdict = result["comparison"].get("verdict", "unknown")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    return {
        "prompt_count": len(results),
        "total_evaluator_calls": sum(int(result.get("evaluator_calls", 0)) for result in results),
        "total_ascr_insertions": sum(int(result.get("ascr_insertions", 0)) for result in results),
        "verdicts": verdicts,
        "results": results,
    }


def suite_to_markdown(suite):
    lines = [
        "| Index | Prompt | Start | Evaluator calls | ASCR insertions | Stop reason | Baseline | ASCR | Delta | Verdict |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for index, result in enumerate(suite["results"]):
        comparison = result["comparison"]
        summary = result["ascr_summary"]
        prompt_text = result["prompt"].replace("|", "\\|")
        start_mode = result["ascr_start_mode"]
        evaluator_calls = result["evaluator_calls"]
        insertions = result["ascr_insertions"]
        stop_reason = summary.get("stop_reason")
        baseline_score = comparison["baseline_score"]
        ascr_score = comparison["ascr_score"]
        delta = comparison["delta"]
        verdict = comparison["verdict"]
        lines.append(f"| {index} | {prompt_text} | {start_mode} | {evaluator_calls} | {insertions} | {stop_reason} | {baseline_score:.3f} | {ascr_score:.3f} | {delta:.3f} | {verdict} |")
    return chr(10).join(lines) + chr(10)


def main(argv=None):
    args = build_parser().parse_args(argv)
    base_config = load_config(args.config)
    prompts = load_prompts(args.prompt, args.prompts_file, args.prompt_limit)
    root = Path(args.output_dir) / datetime.utcnow().strftime("showo_ascr-%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, prompt in enumerate(prompts):
        prompt_root = prompt_run_dir(root, prompt, index, len(prompts))
        prompt_root.mkdir(parents=True, exist_ok=True)
        result, result_path, markdown_path = run_prompt_comparison(base_config, prompt, prompt_root, args)
        result["result_path"] = str(result_path)
        result["markdown_path"] = str(markdown_path)
        results.append(result)
    if len(results) == 1:
        print(json.dumps({"result_path": results[0]["result_path"], "markdown_path": results[0]["markdown_path"], "comparison": results[0]["comparison"]}, indent=2, sort_keys=True))
    else:
        suite = build_suite(results)
        suite_path = root / "suite.json"
        suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
        suite_md_path = root / "suite.md"
        suite_md_path.write_text(suite_to_markdown(suite), encoding="utf-8")
        print(json.dumps({"suite_path": str(suite_path), "suite_markdown_path": str(suite_md_path), "prompt_count": len(results), "total_ascr_insertions": suite["total_ascr_insertions"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

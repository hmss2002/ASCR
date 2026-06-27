import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ascr.analysis.token_locality import diff_energy_grid_from_paths, summarise_locality, write_heatmap_ppm, write_json
from ascr.core.config import load_config
from ascr.corruption.vq_corruptor import corrupt_vq_ids, token_indices_to_cell_labels
from ascr.generators.lumina_native import LuminaNativeEngine


DEFAULT_CORRUPTION_TYPES = ["block_2x2_random_replace", "block_4x4_random_replace", "local_shuffle_4x4"]
DEFAULT_ANALYSIS_GRIDS = [4, 8, 16]


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_prompts(config, args):
    prompts = []
    prompts.extend(args.prompt or [])
    prompt_file = args.prompt_file or config.get("prompt_file")
    if prompt_file:
        for line in Path(prompt_file).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                prompts.append(text)
    prompts.extend(config.get("prompts", []) or [])
    limit = args.limit if args.limit is not None else config.get("limit")
    if limit is not None:
        prompts = prompts[: int(limit)]
    if not prompts:
        raise ValueError("No prompts supplied. Use --prompt, --prompt-file, or config prompts.")
    return prompts


def _config_list(config, key, default):
    value = config.get(key, default)
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _engine_from_config(config, args):
    generator_config = config.get("generator", {})
    return LuminaNativeEngine(
        checkpoint_path=args.checkpoint_path or generator_config.get("checkpoint_path", "models/lumina-dimoo"),
        repo_path=args.repo_path or generator_config.get("repo_path"),
        lora_path=args.lora_path or generator_config.get("lora_path"),
        device=args.device or generator_config.get("device", "cuda"),
        image_size=int(args.image_size or config.get("image_size", generator_config.get("image_size", 1024))),
        token_grid_size=int(args.token_grid_size or config.get("token_grid_size", generator_config.get("token_grid_size", 64))),
        guidance_scale=float(args.guidance_scale or generator_config.get("guidance_scale", 4.0)),
        generation_timesteps=int(args.generation_timesteps or generator_config.get("generation_timesteps", 64)),
        temperature=float(args.temperature or generator_config.get("temperature", 1.0)),
    )


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    return str(path)


def run_probe(args):
    config = load_config(args.config) if args.config else {}
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/stage3_self_corrupt/locality_probe"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(config, args)
    corruption_types = args.corruption_type or _config_list(config, "corruption_types", DEFAULT_CORRUPTION_TYPES)
    analysis_grids = [int(value) for value in (args.analysis_grid or _config_list(config, "analysis_grids", DEFAULT_ANALYSIS_GRIDS))]
    seed = int(args.seed if args.seed is not None else config.get("seed", 1234))
    engine = _engine_from_config(config, args)
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        clean_vq_ids = engine.generate(prompt, seed=seed + prompt_index)
        clean_dir = output_dir / "images" / f"p{prompt_index:04d}"
        clean_path = clean_dir / "clean.png"
        engine.decode_to(clean_vq_ids, clean_path)
        for corruption_index, corruption_type in enumerate(corruption_types):
            sample_id = f"p{prompt_index:04d}_c{corruption_index:03d}"
            result = corrupt_vq_ids(
                clean_vq_ids,
                token_grid_size=engine.token_grid_size,
                corruption_type=corruption_type,
                seed=seed + prompt_index * 1000 + corruption_index,
            )
            sample_dir = output_dir / "images" / sample_id
            corrupted_path = sample_dir / "corrupted.png"
            engine.decode_to(result.corrupted_vq_ids, corrupted_path)
            token_dir = output_dir / "tokens"
            clean_token_path = token_dir / f"{sample_id}_clean_vq_ids.json"
            corrupted_token_path = token_dir / f"{sample_id}_corrupted_vq_ids.json"
            write_json(clean_token_path, result.clean_vq_ids)
            write_json(corrupted_token_path, result.corrupted_vq_ids)
            metrics = []
            for grid_size in analysis_grids:
                energy = diff_energy_grid_from_paths(clean_path, corrupted_path, grid_size=grid_size)
                heatmap_path = output_dir / "heatmaps" / f"{sample_id}_grid{grid_size}.ppm"
                write_heatmap_ppm(energy, heatmap_path)
                summary = summarise_locality(
                    energy,
                    selected_indices=result.selected_indices,
                    token_grid_size=engine.token_grid_size,
                )
                summary["heatmap"] = str(heatmap_path)
                metrics.append(summary)
            row = {
                "schema_version": "ascr.stage3.token_locality_probe.row.v1",
                "created_at_utc": _created_at(),
                "sample_id": sample_id,
                "prompt": prompt,
                "clean_vq_ids_path": str(clean_token_path),
                "corrupted_vq_ids_path": str(corrupted_token_path),
                "clean_image": str(clean_path),
                "corrupted_image": str(corrupted_path),
                "corruption": result.to_metadata(),
                "coarse_labels_4x4": token_indices_to_cell_labels(result.selected_indices, engine.token_grid_size, 4),
                "coarse_labels_8x8": token_indices_to_cell_labels(result.selected_indices, engine.token_grid_size, 8),
                "coarse_labels_16x16": token_indices_to_cell_labels(result.selected_indices, engine.token_grid_size, 16),
                "metrics": metrics,
            }
            rows.append(row)
    summary = {
        "schema_version": "ascr.stage3.token_locality_probe.summary.v1",
        "created_at_utc": _created_at(),
        "output_dir": str(output_dir),
        "prompt_count": len(prompts),
        "row_count": len(rows),
        "corruption_types": list(corruption_types),
        "analysis_grids": analysis_grids,
        "image_size": engine.image_size,
        "token_grid_size": engine.token_grid_size,
    }
    write_jsonl(output_dir / "manifest.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Probe whether Lumina VQ-token corruption has local decoded-image effects.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--corruption-type", action="append", default=None)
    parser.add_argument("--analysis-grid", action="append", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repo-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--token-grid-size", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--generation-timesteps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    return parser


def main(argv=None):
    summary = run_probe(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

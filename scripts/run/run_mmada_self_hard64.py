#!/usr/bin/env python3
"""Resident Hard64 runner for the MMaDA-8B self-selection Stage-1 task.

Loads ONE MMaDA-8B (shared by generator + self-evaluator) per process and then
loops over its shard of the Hard64 prompts. For each prompt it records both the
baseline image (MMaDA's initial generation, no revision) and the self-revised
final image (closed loop where MMaDA reopens its own low-confidence tokens).

Sharding: process handles prompt indices where ``i % NUM_WORKERS == WORKER_ID``.
Outputs per-prompt PNGs under ``OUT_ROOT/{baseline,self}`` and JSON records under
``OUT_ROOT/records`` so a later merge can build manifests for the Gemini judge.
"""

import json
import os
import sys
from pathlib import Path


def _to_png(src, dst):
    from PIL import Image
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst, format="PNG")
    return str(dst)


def main():
    sys.path.insert(0, ".")
    from ascr.core.config import load_config
    from ascr.core.loop import run_config_from_mapping
    from ascr.core.loop_direct import DirectTokenReopenLoop
    from ascr.evaluators.registry import build_evaluator
    from ascr.generators.registry import build_generator
    from ascr.revision.selector import DirectTokenReopeningSelector
    from ascr.cli.run_stage1_mmada_self import _attach_shared_engine

    config_path = os.environ.get("CONFIG", "configs/stage1/mmada/stage1_mmada8b_self_direct_token.yaml")
    prompts_file = os.environ.get("PROMPTS_FILE", "configs/benchmarks/prompts/t2i_compbench_hard64.txt")
    out_root = Path(os.environ.get("OUT_ROOT", "outputs/mmada_self_hard64"))
    worker_id = int(os.environ.get("WORKER_ID", "0"))
    num_workers = int(os.environ.get("NUM_WORKERS", "1"))
    max_iters = os.environ.get("MAX_ITERS")
    limit = os.environ.get("PROMPT_LIMIT")

    config = load_config(config_path)
    if max_iters:
        config["max_iterations"] = int(max_iters)
    token_grid_size = int(config.get("token_grid_size", 32))
    generator_config = dict(config)
    generator_config["token_grid_size"] = token_grid_size
    generator_config["image_size"] = int(config.get("image_size", 512))

    generator = build_generator(config.get("generator", {}).get("name", "mmada"), generator_config)
    evaluator = build_evaluator(config.get("evaluator", {}).get("name", "mmada_self"), config)
    shared = _attach_shared_engine(generator, evaluator)
    selector = DirectTokenReopeningSelector(
        token_grid_size=token_grid_size,
        select_grid_size=int(config.get("select_grid_size", token_grid_size)),
        dilation=int(config.get("dilation", 0)),
    )
    run_config = run_config_from_mapping(config)
    loop = DirectTokenReopenLoop(generator, evaluator, selector, run_config, label_step=int(config.get("label_step", 4)))

    prompts = [ln.strip() for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
    if limit:
        prompts = prompts[:int(limit)]

    records_dir = out_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    assigned = [(i, p) for i, p in enumerate(prompts) if i % num_workers == worker_id]
    print(f"[worker {worker_id}/{num_workers}] shared_engine={shared} assigned={len(assigned)} prompts", flush=True)

    for idx, prompt in assigned:
        rec_path = records_dir / f"p{idx:03d}.json"
        if rec_path.exists():
            print(f"[worker {worker_id}] skip p{idx:03d} (done)", flush=True)
            continue
        run_dir = out_root / "runs" / f"p{idx:03d}"
        loop.config.output_dir = str(run_dir)
        try:
            summary = loop.run(prompt, project_root=Path.cwd())
        except Exception as exc:
            rec_path.write_text(json.dumps({"prompt": prompt, "idx": idx, "error": str(exc)}))
            print(f"[worker {worker_id}] p{idx:03d} ERROR {exc}", flush=True)
            continue
        baseline_png = _to_png(summary["initial_decoded_image"], out_root / "baseline" / f"p{idx:03d}.png")
        self_png = _to_png(summary["final_decoded_image"], out_root / "self" / f"p{idx:03d}.png")
        rec_path.write_text(json.dumps({
            "prompt": prompt,
            "idx": idx,
            "baseline_image": baseline_png,
            "final_image": self_png,
            "revisions": len(summary.get("revision_records", [])),
            "stop_reason": summary.get("stop_reason"),
            "iterations_recorded": summary.get("iterations_recorded"),
        }))
        print(f"[worker {worker_id}] p{idx:03d} done revisions={len(summary.get('revision_records', []))} stop={summary.get('stop_reason')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

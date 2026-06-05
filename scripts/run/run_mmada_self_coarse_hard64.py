#!/usr/bin/env python3
"""Resident Hard64 runner for the MMaDA-8B COARSE (4x4) self-selection Stage-1 task.

Phase-9 counterpart to ``run_mmada_self_hard64.py``. Loads ONE MMaDA-8B (shared by
generator + coarse self-evaluator) per process and loops over its shard of the
Hard64 prompts. MMaDA judges its own image at a 4x4 grid (A1..D4), the chosen
coarse cells are projected+dilated to the 32x32 token grid, and those tokens are
reopened — the *original* ASCR coarse-then-dilate strategy, but with MMaDA's own
MMU understanding replacing the Qwen-9B selector.

Sharding is GLOBAL across nodes: with ``NODE_COUNT`` nodes each running
``NUM_WORKERS`` GPU workers, process ``(NODE_INDEX, WORKER_ID)`` handles prompt
indices where ``i % (NUM_WORKERS*NODE_COUNT) == NODE_INDEX*NUM_WORKERS + WORKER_ID``.

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
    from ascr.core.loop import ASCRLoop, run_config_from_mapping
    from ascr.evaluators.registry import build_evaluator
    from ascr.generators.registry import build_generator
    from ascr.revision.selector import GridSemanticReopeningSelector
    from ascr.cli.run_stage1_mmada_self_coarse import _attach_shared_engine

    config_path = os.environ.get("CONFIG", "configs/stage1/mmada/stage1_mmada8b_self_coarse.yaml")
    prompts_file = os.environ.get("PROMPTS_FILE", "configs/benchmarks/prompts/t2i_compbench_hard64.txt")
    out_root = Path(os.environ.get("OUT_ROOT", "outputs/mmada_self_coarse_hard64"))
    worker_id = int(os.environ.get("WORKER_ID", "0"))
    num_workers = int(os.environ.get("NUM_WORKERS", "1"))
    node_index = int(os.environ.get("NODE_INDEX", "0"))
    node_count = int(os.environ.get("NODE_COUNT", "1"))
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
    evaluator = build_evaluator(config.get("evaluator", {}).get("name", "mmada_self_coarse"), config)
    shared = _attach_shared_engine(generator, evaluator)
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get("coarse_grid_size", config.get("selector", {}).get("coarse_grid_size", 4))),
        token_grid_size=token_grid_size,
        dilation=int(config.get("dilation", config.get("selector", {}).get("dilation", 1))),
    )
    run_config = run_config_from_mapping(config)
    loop = ASCRLoop(generator, evaluator, selector, run_config)

    prompts = [ln.strip() for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
    if limit:
        prompts = prompts[:int(limit)]

    records_dir = out_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    global_workers = num_workers * node_count
    global_id = node_index * num_workers + worker_id
    assigned = [(i, p) for i, p in enumerate(prompts) if i % global_workers == global_id]
    print(f"[node {node_index}/{node_count} worker {worker_id}/{num_workers} -> global {global_id}/{global_workers}] "
          f"shared_engine={shared} assigned={len(assigned)} prompts", flush=True)

    for idx, prompt in assigned:
        rec_path = records_dir / f"p{idx:03d}.json"
        if rec_path.exists():
            print(f"[global {global_id}] skip p{idx:03d} (done)", flush=True)
            continue
        run_dir = out_root / "runs" / f"p{idx:03d}"
        loop.config.output_dir = str(run_dir)
        try:
            summary = loop.run(prompt, project_root=Path.cwd())
        except Exception as exc:
            rec_path.write_text(json.dumps({"prompt": prompt, "idx": idx, "error": str(exc)}))
            print(f"[global {global_id}] p{idx:03d} ERROR {exc}", flush=True)
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
        print(f"[global {global_id}] p{idx:03d} done revisions={len(summary.get('revision_records', []))} "
              f"stop={summary.get('stop_reason')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resident Hard64 runner for the MMaDA-8B generator + Qwen3.5-9B coarse selector.

This reproduces the *original* ASCR Stage-1 coarse pipeline (4x4 evaluate -> dilation
-> 32x32 token reopen) but swaps ONLY the base generator from Show-o to MMaDA-8B; the
selector remains the Qwen3.5-9B VLM. Because MMaDA (transformers 4.46) and Qwen
(transformers 5.2.dev) cannot share a process, this MMaDA worker runs in ``.venv-mmada``
and delegates evaluation to a paired ``qwen_eval_server.py`` over an IPC dir.

Global sharding across pairs/nodes: with ``NODE_COUNT`` nodes each running
``NUM_PAIRS`` (MMaDA-worker, Qwen-server) pairs, this process (NODE_INDEX, PAIR_ID)
handles prompt indices where ``i % (NUM_PAIRS*NODE_COUNT) == NODE_INDEX*NUM_PAIRS + PAIR_ID``.
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
    from ascr.evaluators.remote_eval import RemoteFileEvaluator
    from ascr.generators.registry import build_generator
    from ascr.revision.selector import GridSemanticReopeningSelector

    config_path = os.environ.get("CONFIG", "configs/stage1_mmada8b_qwen9b_coarse.yaml")
    prompts_file = os.environ.get("PROMPTS_FILE", "configs/prompts/t2i_compbench_hard64.txt")
    out_root = Path(os.environ.get("OUT_ROOT", "outputs/mmada_qwen_coarse_hard64"))
    ipc_dir = Path(os.environ["IPC_DIR"])
    pair_id = int(os.environ.get("PAIR_ID", "0"))
    num_pairs = int(os.environ.get("NUM_PAIRS", "1"))
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
    evaluator = RemoteFileEvaluator(ipc_dir, grid_size=int(config.get("coarse_grid_size", 4)))
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get("coarse_grid_size", config.get("selector", {}).get("coarse_grid_size", 4))),
        token_grid_size=token_grid_size,
        dilation=int(config.get("dilation", config.get("selector", {}).get("dilation", 1))),
    )
    run_config = run_config_from_mapping(config)
    loop = ASCRLoop(generator, evaluator, selector, run_config)

    print(f"[pair {pair_id}] waiting for qwen server at {ipc_dir} ...", flush=True)
    if not evaluator.wait_for_server():
        print(f"[pair {pair_id}] ERROR: qwen server never became ready", flush=True)
        return 1
    print(f"[pair {pair_id}] qwen server ready", flush=True)

    prompts = [ln.strip() for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
    if limit:
        prompts = prompts[:int(limit)]

    records_dir = out_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    global_workers = num_pairs * node_count
    global_id = node_index * num_pairs + pair_id
    assigned = [(i, p) for i, p in enumerate(prompts) if i % global_workers == global_id]
    print(f"[pair {pair_id} -> global {global_id}/{global_workers}] assigned={len(assigned)} prompts", flush=True)

    try:
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
    finally:
        evaluator.stop_server()
        print(f"[pair {pair_id}] sent stop to qwen server", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

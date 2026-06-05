#!/usr/bin/env python3
"""Resident Hard64 runner: Lumina-DiMOO generator + Qwen3.5-9B coarse selector.

Reproduces the *original* ASCR Stage-1 coarse pipeline (4x4 evaluate -> dilation ->
project to the token grid -> reopen) but with Lumina-DiMOO as the base discrete-
diffusion generator and the Qwen3.5-9B VLM as the external selector (same selector
as the Phase-11 Show-o / MMaDA arms).  Lumina pins transformers 4.46.2 and cannot
share a process with Qwen (transformers 5.2.dev), so this Lumina worker runs in
``.venv-lumina`` and delegates evaluation to a paired ``qwen_eval_server.py`` over
an IPC dir.

At 1024x1024 the Lumina token grid is 64x64 = 4096 tokens; coarse 4x4 cells project
to 16x16-token blocks.  Reopen = set selected baseline VQ codes back to MASK and
re-run Lumina's masked-diffusion sampling on only those positions.

Global sharding: with NODE_COUNT nodes each running NUM_PAIRS (Lumina, Qwen) pairs,
process (NODE_INDEX, PAIR_ID) handles prompt indices i where
``i % (NUM_PAIRS*NODE_COUNT) == NODE_INDEX*NUM_PAIRS + PAIR_ID``.
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
    from ascr.generators.lumina_dimoo import LuminaAdapter
    from ascr.revision.selector import GridSemanticReopeningSelector

    config_path = os.environ.get("CONFIG", "configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml")
    prompts_file = os.environ.get("PROMPTS_FILE", "configs/benchmarks/prompts/t2i_compbench_hard64.txt")
    out_root = Path(os.environ.get("OUT_ROOT", "outputs/lumina_qwen_coarse_hard64"))
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
    token_grid_size = int(config.get("token_grid_size", 64))
    gen_cfg = config.get("generator", {})

    generator = LuminaAdapter(
        checkpoint_path=gen_cfg.get("checkpoint_path", "models/lumina-dimoo"),
        repo_path=gen_cfg.get("repo_path", "third_party/Lumina-DiMOO"),
        device=gen_cfg.get("device", "cuda"),
        token_grid_size=token_grid_size,
        image_size=int(config.get("image_size", 1024)),
        guidance_scale=float(gen_cfg.get("guidance_scale", 4.0)),
        generation_timesteps=int(gen_cfg.get("generation_timesteps", 64)),
        temperature=float(gen_cfg.get("temperature", 1.0)),
        seed=int(config.get("seed", gen_cfg.get("seed", 1234))),
    )
    evaluator = RemoteFileEvaluator(ipc_dir, grid_size=int(config.get("coarse_grid_size", 4)))
    selector = GridSemanticReopeningSelector(
        coarse_grid_size=int(config.get("coarse_grid_size", 4)),
        token_grid_size=token_grid_size,
        dilation=int(config.get("dilation", 1)),
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

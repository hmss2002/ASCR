#!/usr/bin/env python3
"""Resident Qwen3.5-9B evaluation server for the MMaDA+Qwen coarse Stage-1 pipeline.

Loads the real :class:`QwenVLEvaluator` ONCE (transformers 5.2.dev fork) and then
serves semantic-evaluation requests written by ``ascr.evaluators.remote_eval`` from a
shared IPC directory. Runs in ``.venv-qwen36`` on its own GPU, paired with a MMaDA
worker that runs in ``.venv-mmada`` on a sibling GPU. See ``remote_eval.py`` for the
file protocol.
"""

import json
import os
import sys
import time
from pathlib import Path


def main():
    sys.path.insert(0, ".")
    from ascr.core.config import load_config
    from ascr.evaluators.registry import build_evaluator

    config_path = os.environ.get("CONFIG", "configs/stage1_mmada8b_qwen9b_coarse.yaml")
    ipc_dir = Path(os.environ["IPC_DIR"])
    idle_timeout = float(os.environ.get("SERVER_IDLE_TIMEOUT", "3600"))
    poll = float(os.environ.get("SERVER_POLL", "0.5"))

    requests_dir = ipc_dir / "requests"
    responses_dir = ipc_dir / "responses"
    processed_dir = ipc_dir / "processed"
    for d in (requests_dir, responses_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)
    stop_file = ipc_dir / "server_stop"
    ready_file = ipc_dir / "server_ready"

    config = load_config(config_path)
    evaluator = build_evaluator(config.get("evaluator", {}).get("name", "qwen_vl"), config)
    # Force a load now so the worker only starts once the 9B is resident.
    try:
        evaluator._load()
    except Exception as exc:
        print(f"[qwen-server] warm load failed (will lazy-load on first request): {exc}", flush=True)
    ready_file.write_text("ready")
    print(f"[qwen-server] ready ipc={ipc_dir}", flush=True)

    last_activity = time.time()
    def _write_atomic(path, text):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    while True:
        pending = sorted(p for p in requests_dir.glob("r*.json") if not p.name.endswith(".tmp"))
        if not pending:
            if stop_file.exists():
                print("[qwen-server] stop sentinel seen, exiting", flush=True)
                break
            if time.time() - last_activity > idle_timeout:
                print("[qwen-server] idle timeout, exiting", flush=True)
                break
            time.sleep(poll)
            continue
        for req_path in pending:
            resp_path = responses_dir / req_path.name
            if resp_path.exists():
                req_path.replace(processed_dir / req_path.name)
                continue
            try:
                req = json.loads(req_path.read_text())
            except Exception:
                time.sleep(poll)
                continue
            try:
                evaluation = evaluator.evaluate(
                    req["original_prompt"], req["grid_image_path"], req.get("iteration", 0),
                    req.get("current_prompt"),
                )
                payload = evaluation.to_dict()
            except Exception as exc:
                payload = {"should_abstain": True, "has_error": False,
                           "summary": f"qwen-server evaluate error: {exc}", "parser_error": str(exc),
                           "regions": [], "raw": None}
            _write_atomic(resp_path, json.dumps(payload))
            req_path.replace(processed_dir / req_path.name)
            last_activity = time.time()
            n = len(payload.get("regions", []))
            print(f"[qwen-server] {req_path.name} has_error={payload.get('has_error')} regions={n}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

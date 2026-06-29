"""Run one Stage-5 self-corruption ASCR repair loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ascr.core.peft_compat import ensure_transformers_tensor_parallel_compat
from ascr.core.config import load_config
from ascr.corruption.vq_corruptor import corrupt_vq_ids, token_indices_to_cell_labels
from ascr.evaluators.lumina_native import call_native_answer
from ascr.generators.lumina_native import LuminaNativeEngine
from ascr.selectors.mmu_localizer_selector import MMULocalizerSelector
from ascr.training.stage4_mmu_lora import mmu_localization_prompt


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _file_sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_ppm(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r, g, b = rgb
    path.write_text(f"P3\n1 1\n255\n{r} {g} {b}\n", encoding="ascii")
    return str(path)


class MockLuminaEngine:
    token_grid_size = 64
    image_size = 1024

    def generate(self, prompt, seed=0):
        return [int((idx + seed) % 8192) for idx in range(self.token_grid_size * self.token_grid_size)]

    def reopen(self, baseline_vq_ids, selected_indices, prompt, seed=0):
        repaired = list(baseline_vq_ids)
        for row, col in selected_indices:
            index = int(row) * self.token_grid_size + int(col)
            if 0 <= index < len(repaired):
                repaired[index] = (int(repaired[index]) + 17 + int(seed)) % 8192
        return repaired

    def decode_to(self, vq_ids, output_path):
        total = sum(int(value) for value in vq_ids[:32]) % 255
        return _write_ppm(output_path, (total, (total * 3) % 255, (total * 7) % 255))

    def answer_vq_tokens(self, question, vq_ids, max_new_tokens=384):
        return '{"has_error":true,"corrupted_cells_4x4":["A1"]}'


def _engine(config, lora_path=None, mock=False):
    if mock:
        return MockLuminaEngine()
    generator = config.get("generator", {})
    return LuminaNativeEngine(
        checkpoint_path=generator.get("checkpoint_path", config.get("checkpoint_path", "models/lumina-dimoo")),
        repo_path=generator.get("repo_path", config.get("repo_path")),
        lora_path=lora_path or generator.get("lora_path"),
        device=generator.get("device", config.get("device", "cuda")),
        image_size=int(generator.get("image_size", config.get("image_size", 1024))),
        token_grid_size=int(generator.get("token_grid_size", config.get("token_grid_size", 64))),
        generation_timesteps=int(generator.get("generation_timesteps", config.get("generation_timesteps", 64))),
        guidance_scale=float(generator.get("guidance_scale", config.get("guidance_scale", 4.0))),
        temperature=float(generator.get("temperature", config.get("temperature", 1.0))),
    )


def _maybe_attach_lora(engine, lora_path):
    if not lora_path or isinstance(engine, MockLuminaEngine):
        return engine
    if getattr(engine, "lora_path", None) == str(lora_path):
        return engine
    if getattr(engine, "_model", None) is None:
        engine.lora_path = str(lora_path)
        return engine
    ensure_transformers_tensor_parallel_compat()
    from peft import PeftModel

    engine._model = PeftModel.from_pretrained(engine._model, str(lora_path))
    engine._model.eval()
    engine.lora_path = str(lora_path)
    return engine


def _offload_engine_before_mmu(engine, lora_path):
    if not lora_path or isinstance(engine, MockLuminaEngine):
        return False
    if hasattr(engine, "unload"):
        return bool(engine.unload(clear_lora=False))
    return False


def run_stage5_loop(
    prompt,
    output_dir,
    config=None,
    seed=0,
    corruption_type="block_4x4_random_replace",
    grid_size=4,
    max_selected_cells=4,
    lora_path=None,
    mock=False,
):
    config = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_size = int(config.get("grid_size", grid_size))
    max_selected_cells = int(config.get("max_selected_cells", max_selected_cells))
    seed = int(config.get("seed", seed))
    corruption_type = config.get("corruption_type", corruption_type)
    lora_path = lora_path or config.get("lora_path")
    token_grid_size = int(config.get("token_grid_size", (config.get("generator") or {}).get("token_grid_size", 64)))

    share_engine = bool(config.get("share_engine", True))
    gen_engine = _engine(config, lora_path=None, mock=mock)
    mmu_engine = gen_engine if share_engine else _engine(config, lora_path=lora_path, mock=mock)
    clean_vq_ids = gen_engine.generate(prompt, seed=seed)
    clean_path = output_dir / "clean.ppm"
    gen_engine.decode_to(clean_vq_ids, clean_path)
    result = corrupt_vq_ids(
        clean_vq_ids,
        token_grid_size=token_grid_size,
        corruption_type=corruption_type,
        seed=seed + 1,
    )
    corrupted_path = output_dir / "corrupted.ppm"
    gen_engine.decode_to(result.corrupted_vq_ids, corrupted_path)
    target_cells = token_indices_to_cell_labels(result.selected_indices, token_grid_size, grid_size)
    question = mmu_localization_prompt(
        prompt,
        grid_size=grid_size,
        max_selected_cells=max_selected_cells,
        target_schema="localization_cells",
    )
    offloaded_generator_before_mmu = False
    if share_engine and bool(config.get("offload_generator_before_mmu", True)):
        offloaded_generator_before_mmu = _offload_engine_before_mmu(mmu_engine, lora_path)
    mmu_engine = _maybe_attach_lora(mmu_engine, lora_path)
    raw_text, answer_method = call_native_answer(
        mmu_engine,
        question,
        vq_ids=result.corrupted_vq_ids,
        max_new_tokens=int(config.get("max_new_tokens", 384)),
    )
    selector = MMULocalizerSelector(raw_text, grid_size=grid_size, token_grid_size=token_grid_size)
    mask = selector.to_token_mask()
    repaired_vq_ids = gen_engine.reopen(result.corrupted_vq_ids, mask.selected_indices(), prompt, seed=seed + 2)
    repaired_path = output_dir / "repaired.ppm"
    gen_engine.decode_to(repaired_vq_ids, repaired_path)
    predicted_cells = selector.stats()["cells"]
    trace = {
        "schema_version": "ascr.stage5.self_corrupt_loop.v1",
        "created_at_utc": created_at_utc(),
        "prompt": prompt,
        "seed": seed,
        "corruption_type": corruption_type,
        "share_engine": share_engine,
        "offloaded_generator_before_mmu": offloaded_generator_before_mmu,
        "grid_size": grid_size,
        "token_grid_size": token_grid_size,
        "target_cells": target_cells,
        "lora_cells": predicted_cells,
        "mask_stats": selector.stats(),
        "raw_mmu_text": raw_text,
        "answer_method": answer_method,
        "clean_image": str(clean_path),
        "corrupted_image": str(corrupted_path),
        "repaired_image": str(repaired_path),
        "clean_sha256": _file_sha256(clean_path),
        "corrupted_sha256": _file_sha256(corrupted_path),
        "repaired_sha256": _file_sha256(repaired_path),
        "reopen_changed": _file_sha256(corrupted_path) != _file_sha256(repaired_path),
    }
    write_json(output_dir / "trace.json", trace)
    write_json(output_dir / "reopen_mask.json", mask.to_dict())
    write_json(output_dir / "clean_vq_ids.json", clean_vq_ids)
    write_json(output_dir / "corrupted_vq_ids.json", result.corrupted_vq_ids)
    write_json(output_dir / "repaired_vq_ids.json", repaired_vq_ids)
    return trace


def build_parser():
    parser = argparse.ArgumentParser(description="Run one Stage-5 self-corruption ASCR loop.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corruption-type", default="block_4x4_random_replace")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-selected-cells", type=int, default=4)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--mock", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    trace = run_stage5_loop(
        args.prompt,
        args.output_dir,
        config=config,
        seed=args.seed,
        corruption_type=args.corruption_type,
        grid_size=args.grid_size,
        max_selected_cells=args.max_selected_cells,
        lora_path=args.lora_path,
        mock=bool(args.mock or config.get("mock", False)),
    )
    print(json.dumps(trace, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

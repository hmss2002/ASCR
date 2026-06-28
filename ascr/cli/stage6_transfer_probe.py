"""Probe Stage-4 MMU LoRA localization on real generated images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ascr.cli.stage5_self_corrupt_loop import MockLuminaEngine, created_at_utc, write_json
from ascr.core.config import load_config
from ascr.evaluators.lumina_native import call_native_answer
from ascr.generators.lumina_native import LuminaNativeEngine
from ascr.selectors.mmu_localizer_selector import MMULocalizerSelector
from ascr.training.stage4_mmu_lora import mmu_localization_prompt


def read_prompts(path, limit=None):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            rows.append(text)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    return str(path)


def _engine(config, lora_path=None, mock=False):
    if mock:
        return MockLuminaEngine()
    generator = config.get("generator", {})
    return LuminaNativeEngine(
        checkpoint_path=config.get("checkpoint_path", generator.get("checkpoint_path", "models/lumina-dimoo")),
        repo_path=config.get("repo_path", generator.get("repo_path")),
        lora_path=lora_path,
        device=config.get("device", generator.get("device", "cuda")),
        image_size=int(config.get("image_size", generator.get("image_size", 1024))),
        token_grid_size=int(config.get("token_grid_size", generator.get("token_grid_size", 64))),
    )


def run_transfer_probe(args):
    config = load_config(args.config) if args.config else {}
    prompts = read_prompts(args.prompts, limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_size = int(config.get("grid_size", args.grid_size))
    token_grid_size = int(config.get("token_grid_size", args.token_grid_size))
    max_selected_cells = int(config.get("max_selected_cells", args.max_selected_cells))
    gen_engine = _engine(config, mock=args.mock)
    mmu_engine = _engine(config, lora_path=args.lora_path or config.get("lora_path"), mock=args.mock)
    rows = []
    for index, prompt in enumerate(prompts):
        sample_dir = output_dir / f"sample_{index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        vq_ids = gen_engine.generate(prompt, seed=int(args.seed) + index)
        image_path = sample_dir / "generated.ppm"
        gen_engine.decode_to(vq_ids, image_path)
        question = mmu_localization_prompt(
            prompt,
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
            target_schema="localization_cells",
        )
        try:
            raw_text, method = call_native_answer(mmu_engine, question, vq_ids=vq_ids, max_new_tokens=int(args.max_new_tokens))
            selector = MMULocalizerSelector(raw_text, grid_size=grid_size, token_grid_size=token_grid_size)
            cells = selector.stats()["cells"]
            status = "parsed"
            error = None
        except Exception as exc:
            raw_text = ""
            method = None
            cells = []
            status = "error"
            error = f"{exc.__class__.__name__}: {exc}"
        row = {
            "schema_version": "ascr.stage6.transfer_probe.row.v1",
            "created_at_utc": created_at_utc(),
            "sample_index": index,
            "prompt": prompt,
            "image": str(image_path),
            "vq_ids_path": write_json(sample_dir / "vq_ids.json", vq_ids),
            "raw_mmu_text": raw_text,
            "answer_method": method,
            "lora_cells": cells,
            "predicted_cells": cells,
            "status": status,
            "error": error,
        }
        rows.append(row)
    manifest = write_jsonl(output_dir / "manifest.jsonl", rows)
    summary = {
        "schema_version": "ascr.stage6.transfer_probe.summary.v1",
        "created_at_utc": created_at_utc(),
        "prompt_count": len(prompts),
        "row_count": len(rows),
        "parsed_count": sum(1 for row in rows if row["status"] == "parsed"),
        "nonempty_count": sum(1 for row in rows if row.get("lora_cells")),
        "manifest": manifest,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Probe Stage-4 MMU LoRA transfer on real generated images.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--token-grid-size", type=int, default=64)
    parser.add_argument("--max-selected-cells", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    return parser


def main(argv=None):
    summary = run_transfer_probe(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


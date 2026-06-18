import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ascr.core.schemas import safe_parse_semantic_evaluation
from ascr.evaluators.lumina_native import native_eval_prompt
from ascr.training.localizer_model import read_jsonl, resolve_training_image


def _target_from_localization(localization, grid_size=4, max_selected_cells=6):
    payload = localization.get("evaluation", {})
    evaluation = safe_parse_semantic_evaluation(payload, grid_size=grid_size, max_selected_cells=max_selected_cells)
    target = evaluation.to_dict()
    target.pop("raw", None)
    return target


def iter_sft_examples(dataset_path, image_root=None, project_root=None, grid_size=4, max_selected_cells=6, limit=None):
    count = 0
    for row in read_jsonl(dataset_path):
        for localization in row.get("localizations", []) or []:
            prompt = localization.get("prompt", row.get("prompt", ""))
            image_path = resolve_training_image(localization.get("grid_image"), image_root=image_root, project_root=project_root)
            target = _target_from_localization(localization, grid_size=grid_size, max_selected_cells=max_selected_cells)
            yield {
                "sample_id": localization.get("sample_id", row.get("sample_id")),
                "prompt": prompt,
                "image_path": str(image_path) if image_path else None,
                "image_exists": bool(image_path and Path(image_path).exists()),
                "input_text": native_eval_prompt(prompt, grid_size=grid_size, max_selected_cells=max_selected_cells),
                "target_json": target,
                "target_text": json.dumps(target, ensure_ascii=False, sort_keys=True),
                "source": localization.get("source", row.get("source", "")),
                "domain": localization.get("domain", row.get("domain", "")),
            }
            count += 1
            if limit is not None and count >= int(limit):
                return


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def prepare_sft_dataset(
    dataset,
    output_dir,
    image_root=None,
    grid_size=4,
    max_selected_cells=6,
    limit=None,
):
    project_root = Path.cwd()
    examples = list(
        iter_sft_examples(
            dataset,
            image_root=image_root,
            project_root=project_root,
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
            limit=limit,
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / "sft_examples.jsonl"
    write_jsonl(sft_path, examples)
    manifest = {
        "schema_version": "ascr.lumina_native_evaluator_sft_dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": str(dataset),
        "image_root": str(image_root) if image_root else None,
        "example_count": len(examples),
        "missing_images": sum(1 for example in examples if not example["image_exists"]),
        "grid_size": int(grid_size),
        "max_selected_cells": int(max_selected_cells),
        "sft_examples": str(sft_path),
        "training_status": "prepared_only",
        "training_blocker": (
            "LoRA/SFT is not launched until the Lumina-native evaluator audit confirms "
            "an image-conditioned text generation hook."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare Lumina-native evaluator SFT data from Qwen teacher labels.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="outputs/stage2_lumina_native/sft_smoke")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-selected-cells", type=int, default=6)
    parser.add_argument("--mode", choices=["prepare-only", "sft-smoke"], default="prepare-only")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = prepare_sft_dataset(
        args.dataset,
        args.output_dir,
        image_root=args.image_root,
        grid_size=args.grid_size,
        max_selected_cells=args.max_selected_cells,
        limit=args.limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.mode == "sft-smoke":
        raise SystemExit(
            "Lumina-native LoRA/SFT smoke is blocked until the audit confirms an image-conditioned text generation hook."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

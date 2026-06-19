import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ascr.core.schemas import safe_parse_semantic_evaluation
from ascr.distill.teacher import extract_json_object
from ascr.evaluators.lumina_native import call_native_answer, native_eval_prompt
from ascr.generators.lumina_native import LuminaNativeEngine


SCHEMA_TEXT = (
    '{"has_error": boolean, "summary": string, '
    '"regions": [{"cells": [{"label": string}], "reason": string, '
    '"confidence": number, "error_type": string, "action": "reopen"}], '
    '"correction_instruction": string}'
)


def _cell_labels(grid_size):
    labels = []
    for row in range(int(grid_size)):
        for col in range(int(grid_size)):
            labels.append(f"{chr(ord('A') + row)}{col + 1}")
    return ", ".join(labels)


def prompt_variants(prompt, grid_size=4, max_selected_cells=6):
    cells = _cell_labels(grid_size)
    return {
        "ascr_strict_schema": native_eval_prompt(prompt, grid_size=grid_size, max_selected_cells=max_selected_cells),
        "json_first_minimal": (
            "Return JSON only. Start with { and end with }. No markdown. No explanation.\n"
            f"Allowed cells: {cells}. Select at most {int(max_selected_cells)} cells.\n"
            f"Schema: {SCHEMA_TEXT}\n"
            f"Prompt: {prompt}"
        ),
        "abstain_safe_schema": (
            "You are checking whether a generated image matches a text prompt.\n"
            "Output exactly one valid JSON object and no other text.\n"
            "If unsure, output has_error=false with regions=[].\n"
            f"Allowed cells: {cells}. Max cells: {int(max_selected_cells)}.\n"
            f"Required schema: {SCHEMA_TEXT}\n"
            f"Text prompt: {prompt}"
        ),
    }


def _selected_variants(requested, variants):
    if not requested or requested == ["all"]:
        return list(variants.items())
    selected = []
    for name in requested:
        if name not in variants:
            raise ValueError(f"Unknown prompt variant: {name}. Available: {', '.join(sorted(variants))}")
        selected.append((name, variants[name]))
    return selected


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prompt_for_image(prompts, index):
    if len(prompts) == 1:
        return prompts[0]
    return prompts[index]


def run_probe(args, engine=None):
    images = [str(path) for path in args.image]
    prompts = list(args.prompt)
    if len(prompts) not in {1, len(images)}:
        raise ValueError("--prompt must be supplied once or once per --image")
    if engine is None:
        engine = LuminaNativeEngine(
            checkpoint_path=args.checkpoint_path,
            repo_path=args.repo_path,
            lora_path=args.lora_path,
            device=args.device,
            image_size=args.image_size,
            answer_steps=args.answer_steps,
            answer_block_length=args.answer_block_length,
            answer_temperature=args.answer_temperature,
            answer_cfg_scale=args.answer_cfg_scale,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    parsed_count = 0
    call_error_count = 0
    for image_index, image in enumerate(images):
        prompt = _prompt_for_image(prompts, image_index)
        variants = prompt_variants(prompt, grid_size=args.grid_size, max_selected_cells=args.max_selected_cells)
        for variant_name, question in _selected_variants(args.prompt_variant, variants):
            row = {
                "schema_version": "ascr.lumina_native_json_probe.v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "image": image,
                "prompt": prompt,
                "prompt_variant": variant_name,
                "max_new_tokens": int(args.max_new_tokens),
                "status": "not_run",
            }
            try:
                raw_text, method = call_native_answer(
                    engine,
                    question,
                    image_path=image,
                    max_new_tokens=args.max_new_tokens,
                )
                row["method"] = method
                row["raw_text"] = raw_text
                row["raw_preview"] = raw_text[:500]
                payload = extract_json_object(raw_text)
                parsed = safe_parse_semantic_evaluation(
                    payload,
                    grid_size=args.grid_size,
                    max_selected_cells=args.max_selected_cells,
                )
                row["status"] = "parsed"
                row["parsed"] = parsed.to_dict()
                parsed_count += 1
            except Exception as exc:
                row["status"] = "call_error" if "method" not in row else "abstained_malformed_json"
                row["error_type"] = exc.__class__.__name__
                row["error"] = str(exc)[:1000]
                if row["status"] == "call_error":
                    call_error_count += 1
            rows.append(row)
    row_count = len(rows)
    summary = {
        "schema_version": "ascr.lumina_native_json_probe.summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "row_count": row_count,
        "parsed_count": parsed_count,
        "malformed_count": sum(1 for row in rows if row["status"] == "abstained_malformed_json"),
        "call_error_count": call_error_count,
        "parse_rate": parsed_count / row_count if row_count else 0.0,
        "output_dir": str(output_dir),
        "answer_steps": int(args.answer_steps),
        "answer_block_length": int(args.answer_block_length),
        "answer_temperature": float(args.answer_temperature),
        "answer_cfg_scale": float(args.answer_cfg_scale),
        "lora_path": str(args.lora_path) if args.lora_path else None,
    }
    write_jsonl(output_dir / "probe_rows.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Probe Lumina-native MMU JSON compliance for ASCR SemanticEvaluation.")
    parser.add_argument("--image", action="append", required=True, help="Grid/current image path. Repeat for multiple images.")
    parser.add_argument("--prompt", action="append", required=True, help="Original text prompt. Supply once or once per image.")
    parser.add_argument("--output-dir", default="outputs/stage2_lumina_native/json_probe")
    parser.add_argument("--repo-path", default=None)
    parser.add_argument("--checkpoint-path", default="models/lumina-dimoo")
    parser.add_argument("--lora-path", default=None, help="Optional PEFT LoRA adapter path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-selected-cells", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--answer-steps", type=int, default=64)
    parser.add_argument("--answer-block-length", type=int, default=128)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--answer-cfg-scale", type=float, default=0.0)
    parser.add_argument("--prompt-variant", action="append", default=None, help="Prompt variant name or all.")
    return parser


def main(argv=None):
    summary = run_probe(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage-4 native Lumina MMU/LoRA utilities for self-corruption repair."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random

from ascr.core.schemas import GridCell, safe_parse_semantic_evaluation
from ascr.distill.teacher import extract_json_object
from ascr.evaluators.lumina_native import call_native_answer
from ascr.generators.lumina_native import LuminaNativeEngine
from ascr.training.stage3_selectors import (
    evaluate_predictions,
    read_jsonl,
    resolve_path,
    target_cells,
    write_json,
    write_jsonl,
)


SFT_ROW_SCHEMA = "ascr.stage4.mmu_lora_sft.row.v1"
PROBE_ROW_SCHEMA = "ascr.stage4.mmu_localization_probe.row.v1"


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cell_labels(grid_size):
    return [GridCell(row, col).to_label() for row in range(int(grid_size)) for col in range(int(grid_size))]


def mmu_localization_prompt(prompt, grid_size=16, max_selected_cells=16):
    cells = ", ".join(_cell_labels(grid_size))
    return (
        "You are the ASCR native Lumina-MMU corruption localizer.\n"
        "The input image is a generated image whose internal VQ image tokens may "
        "have been artificially corrupted in a small local region.\n"
        "Use the text prompt and the image evidence to identify corrupted grid cells.\n"
        "Return exactly one compact JSON object. No markdown. No analysis.\n"
        "Schema: {\"has_error\": boolean, \"summary\": string, "
        "\"regions\": [{\"cells\": [{\"label\": string}], \"reason\": string, "
        "\"confidence\": number, \"error_type\": \"self_corruption\", "
        "\"action\": \"reopen\"}], \"correction_instruction\": string}\n"
        f"Allowed {int(grid_size)}x{int(grid_size)} grid cells: {cells}.\n"
        f"Use at most {int(max_selected_cells)} selected cells.\n"
        "If no corrupted region is visible, output has_error=false and regions=[].\n"
        f"Original prompt: {prompt}"
    )


def repair_target_payload(row, grid_size=16, max_selected_cells=16):
    labels = target_cells(row, grid_size)
    labels = labels[: int(max_selected_cells)]
    cells = [{"label": label} for label in labels]
    return {
        "has_error": bool(cells),
        "summary": f"Known self-corruption region for {row.get('corruption_type', 'unknown')}.",
        "regions": [
            {
                "cells": cells,
                "reason": "self-supervised token corruption target",
                "confidence": 1.0,
                "error_type": "self_corruption",
                "action": "reopen",
            }
        ] if cells else [],
        "correction_instruction": "Reopen the selected corrupted image-token region.",
    }


def _split_indices(rows, eval_mode="holdout", train_ratio=0.75, seed=0):
    indices = list(range(len(rows)))
    if eval_mode == "resubstitution":
        return indices, indices
    if eval_mode != "holdout":
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    if len(indices) <= 1:
        return indices, indices
    rng = random.Random(int(seed))
    by_type = {}
    for index, row in enumerate(rows):
        by_type.setdefault(row.get("corruption_type", "unknown"), []).append(index)
    train_indices = []
    eval_indices = []
    for group in by_type.values():
        group = list(group)
        rng.shuffle(group)
        if len(group) == 1:
            train_indices.extend(group)
            continue
        train_count = int(len(group) * float(train_ratio))
        train_count = max(1, min(len(group) - 1, train_count))
        train_indices.extend(group[:train_count])
        eval_indices.extend(group[train_count:])
    return sorted(train_indices), sorted(eval_indices or train_indices)


def _normalised_path(raw_path, project_root=None):
    if not raw_path:
        return None, False
    resolved = resolve_path(raw_path, project_root=project_root)
    return str(resolved), resolved.exists()


def sft_example_from_row(row, grid_size=16, max_selected_cells=16, project_root=None, split=None):
    image_path, image_exists = _normalised_path(row.get("corrupted_image"), project_root=project_root)
    vq_ids_path, vq_ids_exists = _normalised_path(row.get("corrupted_vq_ids_path"), project_root=project_root)
    target = repair_target_payload(row, grid_size=grid_size, max_selected_cells=max_selected_cells)
    full_targets = target_cells(row, grid_size)
    return {
        "schema_version": SFT_ROW_SCHEMA,
        "sample_id": row.get("sample_id"),
        "split": split,
        "prompt": row.get("prompt", ""),
        "image_path": image_path,
        "image_exists": bool(image_exists),
        "vq_ids_path": vq_ids_path,
        "vq_ids_exists": bool(vq_ids_exists),
        "input_text": mmu_localization_prompt(
            row.get("prompt", ""),
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
        ),
        "target_json": target,
        "target_text": json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "grid_size": int(grid_size),
        "max_selected_cells": int(max_selected_cells),
        "target_cells": full_targets,
        "target_truncated": len(full_targets) > int(max_selected_cells),
        "corruption_type": row.get("corruption_type", "unknown"),
        "token_grid_size": int(row.get("token_grid_size") or 64),
        "image_size": int(row.get("image_size") or 1024),
    }


def prepare_mmu_sft_dataset(
    dataset,
    output_dir,
    grid_size=16,
    max_selected_cells=16,
    train_ratio=0.75,
    seed=0,
    eval_mode="holdout",
    limit=None,
    project_root=None,
):
    rows = read_jsonl(dataset)
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset}")
    train_indices, eval_indices = _split_indices(rows, eval_mode=eval_mode, train_ratio=train_ratio, seed=seed)
    train_set = set(train_indices)
    eval_set = set(eval_indices)
    examples = []
    for index, row in enumerate(rows):
        split = "train" if index in train_set else "eval" if index in eval_set else "unused"
        examples.append(
            sft_example_from_row(
                row,
                grid_size=grid_size,
                max_selected_cells=max_selected_cells,
                project_root=project_root,
                split=split,
            )
        )
    train_examples = [examples[index] for index in train_indices]
    eval_examples = [examples[index] for index in eval_indices]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / "sft_examples.jsonl"
    train_path = output_dir / "train_sft_examples.jsonl"
    eval_path = output_dir / "eval_sft_examples.jsonl"
    split_path = output_dir / "split_manifest.json"
    write_jsonl(sft_path, examples)
    write_jsonl(train_path, train_examples)
    write_jsonl(eval_path, eval_examples)
    split_manifest = {
        "schema_version": "ascr.stage4.mmu_lora_split.v1",
        "dataset": str(dataset),
        "eval_mode": eval_mode,
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "row_count": len(rows),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_sample_ids": [row.get("sample_id") for row in train_examples],
        "eval_sample_ids": [row.get("sample_id") for row in eval_examples],
    }
    write_json(split_path, split_manifest)
    manifest = {
        "schema_version": "ascr.stage4.mmu_lora_sft_manifest.v1",
        "created_at_utc": created_at_utc(),
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "grid_size": int(grid_size),
        "max_selected_cells": int(max_selected_cells),
        "row_count": len(examples),
        "train_rows": len(train_examples),
        "eval_rows": len(eval_examples),
        "missing_images": sum(1 for example in examples if not example["image_exists"]),
        "missing_vq_ids": sum(1 for example in examples if not example["vq_ids_exists"]),
        "sft_examples": str(sft_path),
        "train_sft_examples": str(train_path),
        "eval_sft_examples": str(eval_path),
        "split_manifest": str(split_path),
        "target_schema": "canonical_semantic_evaluation_v1",
        "preferred_training_input": "vq_ids_path",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def sample_ids_from_split_manifest(path, split="eval"):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key = "train_sample_ids" if split == "train" else "eval_sample_ids"
    return {str(value) for value in payload.get(key, [])}


def _selected_cells_from_evaluation(evaluation, grid_size):
    selected = []
    for region in evaluation.actionable_regions():
        for cell in region.cells:
            selected.append(GridCell.from_any(cell, grid_size).to_label())
    return sorted(set(selected))


def run_mmu_localization_probe(
    dataset,
    output_dir,
    grid_size=16,
    max_selected_cells=16,
    top_k=4,
    limit=None,
    sample_ids=None,
    split_manifest=None,
    split="eval",
    use_vq_tokens=True,
    lora_path=None,
    engine=None,
    repo_path="third_party/Lumina-DiMOO",
    checkpoint_path="models/lumina-dimoo",
    device="cuda",
    image_size=1024,
    max_new_tokens=384,
    answer_steps=64,
    answer_block_length=128,
    answer_temperature=0.0,
    answer_cfg_scale=0.0,
):
    rows = read_jsonl(dataset)
    wanted = set(str(value) for value in sample_ids or [])
    if split_manifest:
        wanted |= sample_ids_from_split_manifest(split_manifest, split=split)
    if wanted:
        rows = [row for row in rows if str(row.get("sample_id")) in wanted]
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"No probe rows selected from dataset: {dataset}")
    if engine is None:
        engine = LuminaNativeEngine(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            lora_path=lora_path,
            device=device,
            image_size=image_size,
            answer_steps=answer_steps,
            answer_block_length=answer_block_length,
            answer_temperature=answer_temperature,
            answer_cfg_scale=answer_cfg_scale,
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_rows = []
    prediction_map = {}
    parsed_count = 0
    call_error_count = 0
    malformed_count = 0
    for row in rows:
        sample_id = row.get("sample_id")
        question = mmu_localization_prompt(row.get("prompt", ""), grid_size=grid_size, max_selected_cells=max_selected_cells)
        probe_row = {
            "schema_version": PROBE_ROW_SCHEMA,
            "created_at_utc": created_at_utc(),
            "sample_id": sample_id,
            "prompt": row.get("prompt", ""),
            "corruption_type": row.get("corruption_type", "unknown"),
            "grid_size": int(grid_size),
            "target_cells": target_cells(row, grid_size),
            "use_vq_tokens": bool(use_vq_tokens),
            "lora_path": str(lora_path) if lora_path else None,
            "status": "not_run",
        }
        try:
            if use_vq_tokens:
                vq_ids = json.loads(Path(row["corrupted_vq_ids_path"]).read_text(encoding="utf-8"))
                raw_text, method = call_native_answer(
                    engine,
                    question,
                    vq_ids=vq_ids,
                    max_new_tokens=max_new_tokens,
                )
            else:
                raw_text, method = call_native_answer(
                    engine,
                    question,
                    image_path=row.get("corrupted_image"),
                    max_new_tokens=max_new_tokens,
                )
            probe_row["method"] = method
            probe_row["raw_text"] = raw_text
            payload = extract_json_object(raw_text)
            evaluation = safe_parse_semantic_evaluation(
                payload,
                grid_size=grid_size,
                max_selected_cells=max_selected_cells,
            )
            probe_row["parsed"] = evaluation.to_dict()
            if evaluation.should_abstain:
                probe_row["status"] = "abstained_malformed_json"
                malformed_count += 1
                predicted = []
            else:
                probe_row["status"] = "parsed"
                parsed_count += 1
                predicted = _selected_cells_from_evaluation(evaluation, grid_size)
        except Exception as exc:
            probe_row["status"] = "call_error" if "method" not in probe_row else "abstained_malformed_json"
            probe_row["error_type"] = exc.__class__.__name__
            probe_row["error"] = str(exc)[:1000]
            if probe_row["status"] == "call_error":
                call_error_count += 1
            else:
                malformed_count += 1
            predicted = []
        probe_row["predicted_cells"] = predicted
        prediction_map[str(sample_id)] = predicted
        probe_rows.append(probe_row)
    examples = [
        {
            "sample_id": str(row.get("sample_id")),
            "prompt": row.get("prompt", ""),
            "corruption_type": row.get("corruption_type", "unknown"),
            "target_cells": target_cells(row, grid_size),
        }
        for row in rows
    ]
    metrics, predictions = evaluate_predictions(
        examples,
        prediction_map,
        grid_size=grid_size,
        baseline="lumina_mmu_lora" if lora_path else "lumina_mmu_zero",
        top_k=top_k,
    )
    row_count = len(probe_rows)
    summary = {
        "schema_version": "ascr.stage4.mmu_localization_probe.summary.v1",
        "created_at_utc": created_at_utc(),
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "row_count": row_count,
        "parsed_count": parsed_count,
        "malformed_count": malformed_count,
        "call_error_count": call_error_count,
        "parse_rate": parsed_count / row_count if row_count else 0.0,
        "grid_size": int(grid_size),
        "max_selected_cells": int(max_selected_cells),
        "top_k": int(top_k),
        "use_vq_tokens": bool(use_vq_tokens),
        "split_manifest": str(split_manifest) if split_manifest else None,
        "split": split if split_manifest else None,
        "lora_path": str(lora_path) if lora_path else None,
        "metrics": metrics,
    }
    write_jsonl(output_dir / "probe_rows.jsonl", probe_rows)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "summary.json", summary)
    return summary

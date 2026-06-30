"""Stage-4 native Lumina MMU/LoRA utilities for self-corruption repair."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import time

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
INPUT_MODE_VQ_TOKENS = "vq_tokens"
INPUT_MODE_DECODED_IMAGE = "decoded_image"
INPUT_MODE_BOTH = "both"
TARGET_SCHEMA_SEMANTIC_EVALUATION = "semantic_evaluation"
TARGET_SCHEMA_LOCALIZATION_CELLS = "localization_cells"
TARGET_SCHEMA_REPAIR_CELLS = "repair_cells"
PROMPT_VARIANT_LEGACY_DEFAULT = "default"
PROMPT_VARIANT_MINIMAL_JSON = "minimal_json"
PROMPT_VARIANT_SCHEMA_FIRST = "schema_first"
PROMPT_VARIANT_SCHEMA_EXAMPLE = "schema_example"
PROMPT_VARIANT_DEFAULT = PROMPT_VARIANT_SCHEMA_EXAMPLE
PROMPT_VARIANT_CHOICES = [
    PROMPT_VARIANT_SCHEMA_EXAMPLE,
    PROMPT_VARIANT_LEGACY_DEFAULT,
    PROMPT_VARIANT_MINIMAL_JSON,
    PROMPT_VARIANT_SCHEMA_FIRST,
]


def normalise_input_mode(value, allow_both=False):
    mode = str(value or INPUT_MODE_VQ_TOKENS).strip().lower().replace("-", "_")
    aliases = {
        "vq": INPUT_MODE_VQ_TOKENS,
        "tokens": INPUT_MODE_VQ_TOKENS,
        "vq_token": INPUT_MODE_VQ_TOKENS,
        "vq_tokens": INPUT_MODE_VQ_TOKENS,
        "image": INPUT_MODE_DECODED_IMAGE,
        "decoded": INPUT_MODE_DECODED_IMAGE,
        "decoded_image": INPUT_MODE_DECODED_IMAGE,
        "rgb": INPUT_MODE_DECODED_IMAGE,
        "both": INPUT_MODE_BOTH,
    }
    mode = aliases.get(mode, mode)
    allowed = {INPUT_MODE_VQ_TOKENS, INPUT_MODE_DECODED_IMAGE}
    if allow_both:
        allowed.add(INPUT_MODE_BOTH)
    if mode not in allowed:
        raise ValueError(f"Unsupported Stage-4 input_mode: {value!r}")
    return mode


def normalise_target_schema(value):
    schema = str(value or TARGET_SCHEMA_LOCALIZATION_CELLS).strip().lower().replace("-", "_")
    aliases = {
        "semantic": TARGET_SCHEMA_SEMANTIC_EVALUATION,
        "semantic_evaluation": TARGET_SCHEMA_SEMANTIC_EVALUATION,
        "canonical": TARGET_SCHEMA_SEMANTIC_EVALUATION,
        "canonical_semantic_evaluation": TARGET_SCHEMA_SEMANTIC_EVALUATION,
        "localization": TARGET_SCHEMA_LOCALIZATION_CELLS,
        "localization_cells": TARGET_SCHEMA_LOCALIZATION_CELLS,
        "cell_labels": TARGET_SCHEMA_LOCALIZATION_CELLS,
        "corrupted_cells": TARGET_SCHEMA_LOCALIZATION_CELLS,
        "repair": TARGET_SCHEMA_REPAIR_CELLS,
        "repair_cells": TARGET_SCHEMA_REPAIR_CELLS,
        "error_cells": TARGET_SCHEMA_REPAIR_CELLS,
        "token_repair": TARGET_SCHEMA_REPAIR_CELLS,
    }
    schema = aliases.get(schema, schema)
    if schema not in {TARGET_SCHEMA_SEMANTIC_EVALUATION, TARGET_SCHEMA_LOCALIZATION_CELLS, TARGET_SCHEMA_REPAIR_CELLS}:
        raise ValueError(f"Unsupported Stage-4 target_schema: {value!r}")
    return schema


def normalise_prompt_variant(value):
    variant = str(value or PROMPT_VARIANT_DEFAULT).strip().lower().replace("-", "_")
    aliases = {
        "default": PROMPT_VARIANT_LEGACY_DEFAULT,
        "legacy": PROMPT_VARIANT_LEGACY_DEFAULT,
        "legacy_default": PROMPT_VARIANT_LEGACY_DEFAULT,
        "full": PROMPT_VARIANT_LEGACY_DEFAULT,
        "minimal": PROMPT_VARIANT_MINIMAL_JSON,
        "minimal_json": PROMPT_VARIANT_MINIMAL_JSON,
        "short": PROMPT_VARIANT_MINIMAL_JSON,
        "schema": PROMPT_VARIANT_SCHEMA_FIRST,
        "schema_first": PROMPT_VARIANT_SCHEMA_FIRST,
        "schema_only": PROMPT_VARIANT_SCHEMA_FIRST,
        "example": PROMPT_VARIANT_SCHEMA_EXAMPLE,
        "schema_example": PROMPT_VARIANT_SCHEMA_EXAMPLE,
        "fewshot": PROMPT_VARIANT_SCHEMA_EXAMPLE,
        "few_shot": PROMPT_VARIANT_SCHEMA_EXAMPLE,
    }
    variant = aliases.get(variant, variant)
    allowed = set(PROMPT_VARIANT_CHOICES)
    if variant not in allowed:
        raise ValueError(f"Unsupported Stage-4 prompt_variant: {value!r}")
    return variant


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cell_labels(grid_size):
    return [GridCell(row, col).to_label() for row in range(int(grid_size)) for col in range(int(grid_size))]


def _target_grid_sizes(primary_grid_size):
    primary_grid_size = int(primary_grid_size)
    standard = [4, 8, 16]
    if primary_grid_size in standard:
        return [size for size in standard if size <= primary_grid_size]
    return [primary_grid_size]


def mmu_localization_prompt(
    prompt,
    grid_size=16,
    max_selected_cells=16,
    target_schema=TARGET_SCHEMA_LOCALIZATION_CELLS,
    prompt_variant=PROMPT_VARIANT_DEFAULT,
):
    target_schema = normalise_target_schema(target_schema)
    prompt_variant = normalise_prompt_variant(prompt_variant)
    cells = ", ".join(_cell_labels(grid_size))
    if target_schema == TARGET_SCHEMA_REPAIR_CELLS:
        labels = _cell_labels(grid_size)
        if int(grid_size) >= 8:
            example_payloads = [
                {"cells": ["D4", "D5"]},
                {"cells": ["A1"]},
                {"cells": ["A8", "B8"]},
                {"cells": ["C3", "C4", "D3", "D4"]},
                {"cells": []},
            ]
        else:
            example_payloads = [
                {"cells": labels[: min(2, len(labels))]},
                {"cells": labels[:1]},
                {"cells": []},
            ]
        examples = "\n".join(
            json.dumps(payload, sort_keys=False, separators=(",", ":"))
            for payload in example_payloads
        )
        last_label = labels[-1] if labels else "H8"
        return (
            "You are the ASCR token-state repair cell selector.\n\n"
            "Input: the original text prompt plus the current generated image represented as Lumina VQ tokens.\n\n"
            f"Task: choose which cells on the fixed {int(grid_size)}x{int(grid_size)} repair grid should be reopened because they contain corrupted VQ tokens.\n\n"
            "Return exactly one compact JSON object and nothing else.\n"
            "\n"
            "Schema:\n"
            "{\"cells\": string[]}\n"
            "\n"
            "Examples:\n"
            f"{examples}\n"
            "\n"
            "Rules:\n"
            f"- Use only {int(grid_size)}x{int(grid_size)} cell labels: A1 through {last_label}.\n"
            "- If no repair is needed, return {\"cells\":[]}.\n"
            "- If corrupted tokens touch multiple cells, include every touched cell.\n"
            f"- Sort cells row-major: {cells}.\n"
            "- Do not output any key except \"cells\".\n"
            "- Do not output markdown, prose, confidence, coordinates, explanations, or extra fields.\n"
            f"- Use at most {int(max_selected_cells)} cells.\n"
            "\n"
            "Original prompt:\n"
            f"{prompt}"
        )
    if target_schema == TARGET_SCHEMA_LOCALIZATION_CELLS:
        schema_keys = [
            f'"corrupted_cells_{size}x{size}": string[]'
            for size in _target_grid_sizes(grid_size)
        ]
        schema_text = "{\"has_error\": boolean, " + ", ".join(schema_keys) + "}"
        if prompt_variant == PROMPT_VARIANT_MINIMAL_JSON:
            return (
                "Return JSON only. No prose.\n"
                f"Schema: {schema_text}\n"
                f"Allowed cells: {cells}.\n"
                f"Max {int(max_selected_cells)} cells in corrupted_cells_{int(grid_size)}x{int(grid_size)}.\n"
                f"Prompt: {prompt}"
            )
        if prompt_variant == PROMPT_VARIANT_SCHEMA_FIRST:
            return (
                f"{schema_text}\n"
                "Output exactly the schema above as one compact JSON object.\n"
                "Use true/false booleans and arrays of exact grid labels only.\n"
                "No markdown, no explanation, no alternate keys.\n"
                f"Allowed {int(grid_size)}x{int(grid_size)} labels: {cells}.\n"
                f"Original prompt: {prompt}"
            )
        if prompt_variant == PROMPT_VARIANT_SCHEMA_EXAMPLE:
            first_label = _cell_labels(grid_size)[0]
            example_key = f"corrupted_cells_{int(grid_size)}x{int(grid_size)}"
            example = json.dumps(
                {"has_error": True, example_key: [first_label]},
                sort_keys=True,
                separators=(",", ":"),
            )
            no_error = json.dumps(
                {"has_error": False, example_key: []},
                sort_keys=True,
                separators=(",", ":"),
            )
            return (
                "Return exactly one compact JSON object and nothing else.\n"
                f"Schema: {schema_text}\n"
                f"Positive example: {example}\n"
                f"No-error example: {no_error}\n"
                f"Allowed {int(grid_size)}x{int(grid_size)} labels: {cells}.\n"
                f"Use at most {int(max_selected_cells)} selected cells.\n"
                f"Original prompt: {prompt}"
            )
        return (
            "You are the ASCR native Lumina-MMU corruption localizer.\n"
            "The input image is a generated image whose internal VQ image tokens may "
            "have been artificially corrupted in a small local region.\n"
            "Use the text prompt and the image evidence to identify corrupted grid cells.\n"
            "Return exactly one compact JSON object. No markdown. No analysis.\n"
            "Use exact key names and put only grid-cell labels in the arrays.\n"
            "Do not put cell lists inside correction_instruction.\n"
            f"Schema: {schema_text}\n"
            f"Allowed {int(grid_size)}x{int(grid_size)} grid cells: {cells}.\n"
            f"Use at most {int(max_selected_cells)} selected cells for the {int(grid_size)}x{int(grid_size)} grid.\n"
            "If no corrupted region is visible, output has_error=false and all corrupted_cells arrays as [].\n"
            f"Original prompt: {prompt}"
        )
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


def localization_target_payload(row, grid_size=16, max_selected_cells=16):
    payload = {"has_error": False}
    for size in _target_grid_sizes(grid_size):
        labels = target_cells(row, size)[: int(max_selected_cells)]
        payload[f"corrupted_cells_{size}x{size}"] = labels
        if labels:
            payload["has_error"] = True
    return payload


def repair_cells_target_payload(row, grid_size=8, max_selected_cells=16):
    labels = target_cells(row, grid_size)[: int(max_selected_cells)]
    return {"cells": labels}


def target_payload_text(target, target_schema=TARGET_SCHEMA_LOCALIZATION_CELLS):
    target_schema = normalise_target_schema(target_schema)
    return json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=target_schema != TARGET_SCHEMA_REPAIR_CELLS,
        separators=(",", ":"),
    )


def repair_target_payload(
    row,
    grid_size=16,
    max_selected_cells=16,
    target_schema=TARGET_SCHEMA_SEMANTIC_EVALUATION,
):
    target_schema = normalise_target_schema(target_schema)
    if target_schema == TARGET_SCHEMA_REPAIR_CELLS:
        return repair_cells_target_payload(
            row,
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
        )
    if target_schema == TARGET_SCHEMA_LOCALIZATION_CELLS:
        return localization_target_payload(
            row,
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
        )
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


def _split_indices(rows, eval_mode="holdout", train_ratio=0.75, val_ratio=0.0, seed=0):
    indices = list(range(len(rows)))
    if eval_mode == "resubstitution":
        return indices, indices, indices
    if eval_mode != "holdout":
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    if len(indices) <= 1:
        return indices, [], indices
    if any(row.get("split_group_id") or row.get("source_clean_sample_id") for row in rows):
        groups = {}
        for index, row in enumerate(rows):
            group_id = str(row.get("split_group_id") or row.get("source_clean_sample_id") or row.get("sample_id") or index)
            groups.setdefault(group_id, []).append(index)
        group_ids = sorted(groups)
        rng = random.Random(int(seed))
        rng.shuffle(group_ids)
        train_count = int(len(group_ids) * float(train_ratio))
        train_count = max(1, min(len(group_ids) - 1, train_count))
        remaining = len(group_ids) - train_count
        val_count = int(len(group_ids) * max(0.0, float(val_ratio or 0.0)))
        if val_ratio > 0 and remaining >= 2:
            val_count = max(1, min(remaining - 1, val_count))
        else:
            val_count = 0
        train_groups = set(group_ids[:train_count])
        val_groups = set(group_ids[train_count:train_count + val_count])
        test_groups = set(group_ids[train_count + val_count:])
        train_indices = sorted(index for group in train_groups for index in groups[group])
        val_indices = sorted(index for group in val_groups for index in groups[group])
        test_indices = sorted(index for group in test_groups for index in groups[group])
        if not test_indices:
            test_indices = val_indices or train_indices
        return train_indices, val_indices, test_indices
    train_ratio = float(train_ratio)
    val_ratio = max(0.0, float(val_ratio or 0.0))
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1.0, got {train_ratio + val_ratio}")
    rng = random.Random(int(seed))
    by_type = {}
    for index, row in enumerate(rows):
        by_type.setdefault(row.get("corruption_type", "unknown"), []).append(index)
    train_indices = []
    val_indices = []
    test_indices = []
    for group in by_type.values():
        group = list(group)
        rng.shuffle(group)
        if len(group) == 1:
            train_indices.extend(group)
            continue
        train_count = int(len(group) * train_ratio)
        train_count = max(1, min(len(group) - 1, train_count))
        remaining = len(group) - train_count
        val_count = int(len(group) * val_ratio)
        if val_ratio > 0 and remaining >= 2:
            val_count = max(1, min(remaining - 1, val_count))
        else:
            val_count = 0
        train_indices.extend(group[:train_count])
        val_indices.extend(group[train_count:train_count + val_count])
        test_indices.extend(group[train_count + val_count:])
    if not test_indices:
        test_indices = val_indices or train_indices
    return sorted(train_indices), sorted(val_indices), sorted(test_indices)


def _normalised_path(raw_path, project_root=None):
    if not raw_path:
        return None, False
    resolved = resolve_path(raw_path, project_root=project_root)
    return str(resolved), resolved.exists()


def sft_example_from_row(
    row,
    grid_size=16,
    max_selected_cells=16,
    project_root=None,
    split=None,
    input_mode=INPUT_MODE_VQ_TOKENS,
    target_schema=TARGET_SCHEMA_LOCALIZATION_CELLS,
    prompt_variant=PROMPT_VARIANT_DEFAULT,
):
    input_mode = normalise_input_mode(input_mode)
    target_schema = normalise_target_schema(target_schema)
    prompt_variant = normalise_prompt_variant(prompt_variant)
    image_path, image_exists = _normalised_path(row.get("corrupted_image"), project_root=project_root)
    vq_ids_path, vq_ids_exists = _normalised_path(row.get("corrupted_vq_ids_path"), project_root=project_root)
    target = repair_target_payload(
        row,
        grid_size=grid_size,
        max_selected_cells=max_selected_cells,
        target_schema=target_schema,
    )
    full_targets = target_cells(row, grid_size)
    sample_id = row.get("sample_id")
    return {
        "schema_version": SFT_ROW_SCHEMA,
        "sample_id": sample_id,
        "example_id": f"{sample_id}:{input_mode}" if sample_id is not None else input_mode,
        "split": split,
        "input_mode": input_mode,
        "target_schema": target_schema,
        "prompt_variant": prompt_variant,
        "prompt": row.get("prompt", ""),
        "image_path": image_path,
        "image_exists": bool(image_exists),
        "vq_ids_path": vq_ids_path,
        "vq_ids_exists": bool(vq_ids_exists),
        "input_text": mmu_localization_prompt(
            row.get("prompt", ""),
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
            target_schema=target_schema,
            prompt_variant=prompt_variant,
        ),
        "target_json": target,
        "target_text": target_payload_text(target, target_schema=target_schema),
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
    val_ratio=0.0,
    seed=0,
    eval_mode="holdout",
    limit=None,
    project_root=None,
    input_mode=INPUT_MODE_VQ_TOKENS,
    target_schema=TARGET_SCHEMA_LOCALIZATION_CELLS,
    prompt_variant=PROMPT_VARIANT_DEFAULT,
):
    input_mode = normalise_input_mode(input_mode, allow_both=True)
    target_schema = normalise_target_schema(target_schema)
    prompt_variant = normalise_prompt_variant(prompt_variant)
    input_modes = [INPUT_MODE_VQ_TOKENS, INPUT_MODE_DECODED_IMAGE] if input_mode == INPUT_MODE_BOTH else [input_mode]
    rows = read_jsonl(dataset)
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset}")
    train_indices, val_indices, test_indices = _split_indices(
        rows,
        eval_mode=eval_mode,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    examples = []
    for index, row in enumerate(rows):
        if eval_mode == "resubstitution":
            split = "train_eval"
        elif index in train_set:
            split = "train"
        elif index in val_set:
            split = "val"
        elif index in test_set:
            split = "test"
        else:
            split = "unused"
        for mode in input_modes:
            examples.append(
                sft_example_from_row(
                    row,
                    grid_size=grid_size,
                    max_selected_cells=max_selected_cells,
                    project_root=project_root,
                    split=split,
                    input_mode=mode,
                    target_schema=target_schema,
                    prompt_variant=prompt_variant,
                )
            )
    train_ids = {str(rows[index].get("sample_id")) for index in train_indices}
    val_ids = {str(rows[index].get("sample_id")) for index in val_indices}
    test_ids = {str(rows[index].get("sample_id")) for index in test_indices}
    train_examples = [
        example
        for example in examples
        if str(example.get("sample_id")) in train_ids and example.get("split") in {"train", "train_eval"}
    ]
    val_examples = [
        example
        for example in examples
        if str(example.get("sample_id")) in val_ids and example.get("split") in {"val", "train_eval"}
    ]
    test_examples = [
        example
        for example in examples
        if str(example.get("sample_id")) in test_ids and example.get("split") in {"test", "train_eval"}
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / "sft_examples.jsonl"
    train_path = output_dir / "train_sft_examples.jsonl"
    val_path = output_dir / "val_sft_examples.jsonl"
    test_path = output_dir / "test_sft_examples.jsonl"
    eval_path = output_dir / "eval_sft_examples.jsonl"
    split_path = output_dir / "split_manifest.json"
    write_jsonl(sft_path, examples)
    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)
    write_jsonl(test_path, test_examples)
    write_jsonl(eval_path, test_examples)
    split_manifest = {
        "schema_version": "ascr.stage4.mmu_lora_split.v1",
        "dataset": str(dataset),
        "eval_mode": eval_mode,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio or 0.0),
        "test_ratio": max(0.0, 1.0 - float(train_ratio) - float(val_ratio or 0.0)),
        "seed": int(seed),
        "row_count": len(rows),
        "input_mode": input_mode,
        "input_modes": input_modes,
        "target_schema": target_schema,
        "prompt_variant": prompt_variant,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "eval_indices": test_indices,
        "train_sample_ids": [rows[index].get("sample_id") for index in train_indices],
        "val_sample_ids": [rows[index].get("sample_id") for index in val_indices],
        "test_sample_ids": [rows[index].get("sample_id") for index in test_indices],
        "eval_sample_ids": [rows[index].get("sample_id") for index in test_indices],
    }
    write_json(split_path, split_manifest)
    mode_counts = {
        mode: sum(1 for example in examples if example.get("input_mode") == mode)
        for mode in input_modes
    }
    missing_required_inputs = sum(
        1
        for example in examples
        if (
            example.get("input_mode") == INPUT_MODE_VQ_TOKENS
            and not example.get("vq_ids_exists")
        )
        or (
            example.get("input_mode") == INPUT_MODE_DECODED_IMAGE
            and not example.get("image_exists")
        )
    )
    manifest = {
        "schema_version": "ascr.stage4.mmu_lora_sft_manifest.v1",
        "created_at_utc": created_at_utc(),
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "input_mode": input_mode,
        "input_modes": input_modes,
        "input_mode_counts": mode_counts,
        "target_schema": target_schema,
        "prompt_variant": prompt_variant,
        "grid_size": int(grid_size),
        "max_selected_cells": int(max_selected_cells),
        "source_row_count": len(rows),
        "row_count": len(examples),
        "example_count": len(examples),
        "train_rows": len(train_examples),
        "val_rows": len(val_examples),
        "test_rows": len(test_examples),
        "eval_rows": len(test_examples),
        "missing_images": sum(1 for example in examples if not example["image_exists"]),
        "missing_vq_ids": sum(1 for example in examples if not example["vq_ids_exists"]),
        "missing_required_inputs": missing_required_inputs,
        "sft_examples": str(sft_path),
        "train_sft_examples": str(train_path),
        "val_sft_examples": str(val_path),
        "test_sft_examples": str(test_path),
        "eval_sft_examples": str(eval_path),
        "split_manifest": str(split_path),
        "preferred_training_input": (
            "vq_ids_path"
            if input_mode == INPUT_MODE_VQ_TOKENS
            else "image_path"
            if input_mode == INPUT_MODE_DECODED_IMAGE
            else "mixed"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def sample_ids_from_split_manifest(path, split="eval"):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = str(split or "eval").strip().lower()
    key = {
        "train": "train_sample_ids",
        "val": "val_sample_ids",
        "validation": "val_sample_ids",
        "test": "test_sample_ids",
        "eval": "eval_sample_ids",
    }.get(split, "eval_sample_ids")
    return {str(value) for value in payload.get(key, [])}


def _selected_cells_from_evaluation(evaluation, grid_size):
    selected = []
    for region in evaluation.actionable_regions():
        for cell in region.cells:
            selected.append(GridCell.from_any(cell, grid_size).to_label())
    return sorted(set(selected))


def _bool_from_any(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _cell_label_from_index(index, grid_size):
    index = int(index)
    grid_size = int(grid_size)
    if 0 <= index < grid_size * grid_size:
        row, col = divmod(index, grid_size)
        return GridCell(row, col).to_label()
    if 1 <= index <= grid_size * grid_size:
        row, col = divmod(index - 1, grid_size)
        return GridCell(row, col).to_label()
    raise ValueError(f"Cell index {index} is outside {grid_size}x{grid_size}")


def _flatten_cell_candidates(raw_cells):
    if raw_cells is None:
        return []
    if isinstance(raw_cells, dict):
        values = []
        for key in ("label", "cell", "cells", "grid_cells", "selected_cells", "value"):
            if key in raw_cells:
                values.extend(_flatten_cell_candidates(raw_cells.get(key)))
        if values:
            return values
        return list(raw_cells.values())
    if isinstance(raw_cells, (list, tuple, set)):
        values = []
        for value in raw_cells:
            if isinstance(value, (list, tuple, set, dict)) and not (
                isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(part, int) for part in value)
            ):
                values.extend(_flatten_cell_candidates(value))
            else:
                values.append(value)
        return values
    return [raw_cells]


def _coerce_server_cell_token(value, grid_size):
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"([A-Z])_(\d+)(?:_\d+X\d+)?", text)
    if match:
        return GridCell.from_any(f"{match.group(1)}{match.group(2)}", grid_size).to_label()
    match = re.fullmatch(r"CELL[_-]?(\d)(\d)", text)
    if match:
        row = int(match.group(1)) - 1
        col = int(match.group(2)) - 1
        if 0 <= row < int(grid_size) and 0 <= col < int(grid_size):
            return GridCell(row, col).to_label()
    match = re.fullmatch(r"CELL[_-]?(\d+)", text)
    if match:
        return _cell_label_from_index(int(match.group(1)), grid_size)
    raise ValueError(f"Cannot coerce server cell token: {value!r}")


def _coerce_cell_labels(raw_cells, grid_size, max_selected_cells=None):
    if raw_cells is None:
        return []
    if isinstance(raw_cells, str):
        text = raw_cells.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                raw_cells = json.loads(text)
            except Exception:
                raw_cells = None
        if raw_cells is None:
            upper = text.upper()
            if re.search(r"[A-Z]", upper):
                raw_cells = re.findall(r"[A-Z]\d+|R\d+C\d+", upper)
            else:
                raw_cells = re.findall(r"-?\d+", text)
    elif isinstance(raw_cells, dict):
        raw_cells = _flatten_cell_candidates(raw_cells)
    elif not isinstance(raw_cells, (list, tuple, set)):
        raw_cells = [raw_cells]
    else:
        raw_cells = _flatten_cell_candidates(raw_cells)

    labels = []
    for value in raw_cells:
        try:
            if isinstance(value, int) or (isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip())):
                label = _cell_label_from_index(int(value), grid_size)
            else:
                try:
                    label = GridCell.from_any(value, grid_size).to_label()
                except Exception:
                    label = _coerce_server_cell_token(value, grid_size)
        except Exception:
            continue
        if label not in labels:
            labels.append(label)
        if max_selected_cells is not None and len(labels) >= int(max_selected_cells):
            break
    return labels


def localization_payload_to_semantic_payload(payload, grid_size=16, max_selected_cells=16):
    """Map Stage-4 localization-cell JSON into the ASCR SemanticEvaluation contract."""
    if not isinstance(payload, dict):
        return payload
    if payload.get("regions") is not None:
        return payload
    key = f"corrupted_cells_{int(grid_size)}x{int(grid_size)}"
    source_key = None
    raw_cells = None
    for candidate in (
        key,
        "corrupted_cells",
        "cells",
        "selected_cells",
        "grid_cells",
        "has_cells",
        "has cells",
        "cell_labels",
        "labels",
    ):
        if candidate in payload:
            raw_cells = payload.get(candidate)
            source_key = candidate
            break
    if raw_cells is None:
        loose_values = []
        for key_name, value in payload.items():
            normalised_key = str(key_name).strip().lower().replace(" ", "_")
            if normalised_key.startswith("corrupted_cells_") or normalised_key.startswith("cell"):
                loose_values.extend(_flatten_cell_candidates(value))
                loose_values.append(key_name)
        if loose_values:
            raw_cells = loose_values
            source_key = "loose_payload_keys"
    if raw_cells is None and payload.get("correction_instruction"):
        recovered = _coerce_cell_labels(
            payload.get("correction_instruction"),
            grid_size,
            max_selected_cells=max_selected_cells,
        )
        if recovered:
            raw_cells = recovered
            source_key = "correction_instruction"
    labels = _coerce_cell_labels(
        raw_cells,
        grid_size,
        max_selected_cells=max_selected_cells,
    )
    has_error_raw = payload.get("has_error")
    if has_error_raw is None and "error" in payload:
        has_error_raw = payload.get("error")
    has_error = _bool_from_any(has_error_raw, default=bool(labels))
    if not has_error:
        labels = []
    return {
        "has_error": bool(has_error),
        "summary": str(payload.get("summary", f"Stage-4 localization cells from {source_key or 'empty output'}.")),
        "regions": [
            {
                "cells": [{"label": label} for label in labels],
                "reason": str(payload.get("reason", "self-corruption localization")),
                "confidence": float(payload.get("confidence", 1.0)),
                "error_type": "self_corruption",
                "action": "reopen",
            }
        ] if labels else [],
        "correction_instruction": str(payload.get("correction_instruction", "Reopen the selected corrupted image-token region.")),
    }


def safe_parse_mmu_localization_payload(
    payload,
    grid_size=16,
    max_selected_cells=16,
    require_cells_key=False,
):
    if require_cells_key and isinstance(payload, dict) and payload.get("regions") is None and "cells" not in payload:
        raise ValueError('repair_cells output must include required "cells" key')
    normalised = localization_payload_to_semantic_payload(
        payload,
        grid_size=grid_size,
        max_selected_cells=max_selected_cells,
    )
    return safe_parse_semantic_evaluation(
        normalised,
        grid_size=grid_size,
        max_selected_cells=max_selected_cells,
    ), normalised


def run_mmu_localization_probe(
    dataset,
    output_dir,
    grid_size=16,
    max_selected_cells=16,
    top_k=4,
    limit=None,
    sample_offset=0,
    sample_ids=None,
    split_manifest=None,
    split="eval",
    input_mode=None,
    use_vq_tokens=None,
    target_schema=TARGET_SCHEMA_LOCALIZATION_CELLS,
    prompt_variant=PROMPT_VARIANT_DEFAULT,
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
    target_schema = normalise_target_schema(target_schema)
    prompt_variant = normalise_prompt_variant(prompt_variant)
    if input_mode is None:
        input_mode = INPUT_MODE_VQ_TOKENS if use_vq_tokens is not False else INPUT_MODE_DECODED_IMAGE
    input_mode = normalise_input_mode(input_mode)
    use_vq_tokens = input_mode == INPUT_MODE_VQ_TOKENS
    rows = read_jsonl(dataset)
    wanted = set(str(value) for value in sample_ids or [])
    if split_manifest:
        wanted |= sample_ids_from_split_manifest(split_manifest, split=split)
    if wanted:
        rows = [row for row in rows if str(row.get("sample_id")) in wanted]
    sample_offset = int(sample_offset or 0)
    if sample_offset < 0:
        raise ValueError(f"sample_offset must be >= 0, got {sample_offset}")
    if sample_offset:
        rows = rows[sample_offset:]
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
        question = mmu_localization_prompt(
            row.get("prompt", ""),
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
            target_schema=target_schema,
            prompt_variant=prompt_variant,
        )
        probe_row = {
            "schema_version": PROBE_ROW_SCHEMA,
            "created_at_utc": created_at_utc(),
            "sample_id": sample_id,
            "prompt": row.get("prompt", ""),
            "corruption_type": row.get("corruption_type", "unknown"),
            "grid_size": int(grid_size),
            "input_mode": input_mode,
            "target_schema": target_schema,
            "prompt_variant": prompt_variant,
            "target_cells": target_cells(row, grid_size),
            "use_vq_tokens": bool(use_vq_tokens),
            "lora_path": str(lora_path) if lora_path else None,
            "status": "not_run",
        }
        started = time.perf_counter()
        try:
            if use_vq_tokens:
                vq_ids_path = resolve_path(row["corrupted_vq_ids_path"])
                vq_ids = json.loads(Path(vq_ids_path).read_text(encoding="utf-8"))
                raw_text, method = call_native_answer(
                    engine,
                    question,
                    vq_ids=vq_ids,
                    max_new_tokens=max_new_tokens,
                )
            else:
                image_path = resolve_path(row.get("corrupted_image"))
                raw_text, method = call_native_answer(
                    engine,
                    question,
                    image_path=image_path,
                    max_new_tokens=max_new_tokens,
                )
            probe_row["method"] = method
            probe_row["raw_text"] = raw_text
            payload = extract_json_object(raw_text)
            probe_row["parsed_payload"] = payload
            evaluation, normalised_payload = safe_parse_mmu_localization_payload(
                payload,
                grid_size=grid_size,
                max_selected_cells=max_selected_cells,
                require_cells_key=target_schema == TARGET_SCHEMA_REPAIR_CELLS,
            )
            probe_row["normalised_payload"] = normalised_payload
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
        finally:
            probe_row["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
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
        "sample_offset": sample_offset,
        "parsed_count": parsed_count,
        "malformed_count": malformed_count,
        "call_error_count": call_error_count,
        "parse_rate": parsed_count / row_count if row_count else 0.0,
        "grid_size": int(grid_size),
        "max_selected_cells": int(max_selected_cells),
        "top_k": int(top_k),
        "input_mode": input_mode,
        "target_schema": target_schema,
        "prompt_variant": prompt_variant,
        "use_vq_tokens": bool(use_vq_tokens),
        "max_new_tokens": int(max_new_tokens),
        "answer_steps": int(answer_steps),
        "answer_block_length": int(answer_block_length),
        "answer_temperature": float(answer_temperature),
        "answer_cfg_scale": float(answer_cfg_scale),
        "split_manifest": str(split_manifest) if split_manifest else None,
        "split": split if split_manifest else None,
        "lora_path": str(lora_path) if lora_path else None,
        "mean_latency_ms": sum(row.get("latency_ms", 0.0) for row in probe_rows) / row_count if row_count else None,
        "metrics": metrics,
    }
    write_jsonl(output_dir / "probe_rows.jsonl", probe_rows)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "summary.json", summary)
    return summary

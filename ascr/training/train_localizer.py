import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from ascr.training.localizer_model import (
    cell_labels,
    evaluate_predictions,
    feature_vector,
    iter_localization_examples,
    vector_mean,
)


def split_examples(examples, eval_mode="holdout", train_ratio=0.8, seed=0):
    indices = list(range(len(examples)))
    if eval_mode == "resubstitution":
        return indices, indices
    if eval_mode != "holdout":
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    if len(indices) <= 1:
        return indices, indices
    rng = random.Random(int(seed))
    positives = [index for index, example in enumerate(examples) if example.get("target_cells")]
    negatives = [index for index, example in enumerate(examples) if not example.get("target_cells")]

    def split_group(group):
        if not group:
            return [], []
        shuffled = list(group)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            return shuffled, []
        train_count = int(len(shuffled) * float(train_ratio))
        train_count = max(1, min(len(shuffled) - 1, train_count))
        return shuffled[:train_count], shuffled[train_count:]

    train_pos, eval_pos = split_group(positives)
    train_neg, eval_neg = split_group(negatives)
    train_indices = sorted(train_pos + train_neg)
    eval_indices = sorted(eval_pos + eval_neg)
    if not eval_indices:
        eval_indices = train_indices
    return train_indices, eval_indices


def build_model(train_examples, grid_size=4, prompt_hash_dims=8, max_selected_cells=6):
    labels = cell_labels(grid_size)
    positives_by_cell = {label: [] for label in labels}
    negatives_by_cell = {label: [] for label in labels}
    all_positive = []
    all_negative = []
    missing_images = 0
    for example in train_examples:
        if example.get("missing_image"):
            missing_images += 1
            continue
        target = set(example.get("target_cells", []))
        for label in labels:
            vector = feature_vector(
                example["prompt"],
                example["image_path"],
                label,
                grid_size=grid_size,
                prompt_dims=prompt_hash_dims,
            )
            if label in target:
                positives_by_cell[label].append(vector)
                all_positive.append(vector)
            else:
                negatives_by_cell[label].append(vector)
                all_negative.append(vector)
    global_positive = vector_mean(all_positive) or vector_mean(all_negative)
    global_negative = vector_mean(all_negative) or global_positive
    cells = {}
    for label in labels:
        positive = vector_mean(positives_by_cell[label]) or global_positive
        negative = vector_mean(negatives_by_cell[label]) or global_negative
        bias = 0.0
        if positives_by_cell[label] or negatives_by_cell[label]:
            total = len(positives_by_cell[label]) + len(negatives_by_cell[label])
            prior = len(positives_by_cell[label]) / max(1, total)
            bias = prior - 0.5
        cells[label] = {
            "positive_count": len(positives_by_cell[label]),
            "negative_count": len(negatives_by_cell[label]),
            "positive_centroid": positive,
            "negative_centroid": negative,
            "bias": bias,
        }
    model = {
        "schema_version": "ascr.grid_localizer_v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "grid_size": int(grid_size),
        "prompt_hash_dims": int(prompt_hash_dims),
        "max_selected_cells": int(max_selected_cells),
        "threshold": 0.0,
        "global_positive_centroid": global_positive,
        "global_negative_centroid": global_negative,
        "cells": cells,
        "training_summary": {
            "train_examples": len(train_examples),
            "missing_images": missing_images,
            "positive_labels": len(all_positive),
            "negative_labels": len(all_negative),
        },
    }
    return model


def tune_threshold(model, train_examples):
    candidate_scores = []
    from ascr.training.localizer_model import score_cell

    labels = cell_labels(int(model["grid_size"]))
    for example in train_examples:
        if example.get("missing_image"):
            continue
        for label in labels:
            candidate_scores.append(score_cell(model, example["prompt"], example["image_path"], label))
    if not candidate_scores:
        return 0.0
    candidates = sorted(set(candidate_scores))
    if len(candidates) > 64:
        step = max(1, len(candidates) // 64)
        candidates = candidates[::step]
    best_threshold = candidates[0]
    best_score = -1.0
    for threshold in candidates:
        model["threshold"] = threshold
        metrics, _predictions = evaluate_predictions(train_examples, model)
        score = metrics.get("mean_f1")
        if score is None:
            score = 0.0
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def write_holdout_prompts(path, examples):
    prompts = []
    seen = set()
    for example in examples:
        prompt = str(example.get("prompt", "")).strip()
        if prompt and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    Path(path).write_text("\n".join(prompts) + ("\n" if prompts else ""), encoding="utf-8")


def train_grid_localizer_v0(
    dataset,
    image_root,
    output_dir,
    eval_mode="holdout",
    train_ratio=0.8,
    seed=0,
    grid_size=4,
    prompt_hash_dims=8,
    max_selected_cells=6,
):
    project_root = Path.cwd()
    examples = list(iter_localization_examples(dataset, image_root=image_root, project_root=project_root, grid_size=grid_size))
    if not examples:
        raise ValueError(f"No localization examples found in dataset: {dataset}")
    train_indices, eval_indices = split_examples(examples, eval_mode=eval_mode, train_ratio=train_ratio, seed=seed)
    train_examples = [examples[index] for index in train_indices]
    eval_examples = [examples[index] for index in eval_indices]
    model = build_model(
        train_examples,
        grid_size=grid_size,
        prompt_hash_dims=prompt_hash_dims,
        max_selected_cells=max_selected_cells,
    )
    model["threshold"] = tune_threshold(model, train_examples)
    train_metrics, _train_predictions = evaluate_predictions(train_examples, model)
    eval_metrics, eval_predictions = evaluate_predictions(eval_examples, model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model["dataset"] = str(dataset)
    model["image_root"] = str(image_root) if image_root else None
    model["eval_mode"] = eval_mode
    model["train_ratio"] = float(train_ratio)
    model["seed"] = int(seed)
    write_json(output_dir / "student_model.json", model)
    metrics = {
        "schema_version": "ascr.grid_localizer_v0.metrics",
        "dataset": str(dataset),
        "image_root": str(image_root) if image_root else None,
        "row_count": len(examples),
        "train_rows": len(train_examples),
        "eval_rows": len(eval_examples),
        "eval_mode": eval_mode,
        "train": train_metrics,
        "eval": eval_metrics,
        "missing_images": sum(1 for example in examples if example.get("missing_image")),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", eval_predictions)
    split_manifest = {
        "schema_version": "ascr.grid_localizer_v0.split",
        "dataset": str(dataset),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "row_count": len(examples),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_sample_ids": [examples[index].get("sample_id") for index in train_indices],
        "eval_sample_ids": [examples[index].get("sample_id") for index in eval_indices],
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_holdout_prompts(output_dir / "holdout_prompts.txt", eval_examples)
    return {"model": str(output_dir / "student_model.json"), "metrics": metrics, "output_dir": str(output_dir)}


def build_parser():
    parser = argparse.ArgumentParser(description="Train ASCR student semantic localizers.")
    parser.add_argument("--task", choices=["grid-localizer-v0"], default="grid-localizer-v0")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="outputs/stage2_students/grid_localizer_v0")
    parser.add_argument("--eval-mode", choices=["resubstitution", "holdout"], default="holdout")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--prompt-hash-dims", type=int, default=8)
    parser.add_argument("--max-selected-cells", type=int, default=6)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = train_grid_localizer_v0(
        args.dataset,
        args.image_root,
        args.output_dir,
        eval_mode=args.eval_mode,
        train_ratio=args.train_ratio,
        seed=args.seed,
        grid_size=args.grid_size,
        prompt_hash_dims=args.prompt_hash_dims,
        max_selected_cells=args.max_selected_cells,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

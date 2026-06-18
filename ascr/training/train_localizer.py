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
    predict_cells,
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
                feature_version="v0",
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


def sigmoid(value):
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + pow(2.718281828459045, -value))


def train_logistic_cell(examples, label, grid_size, prompt_hash_dims, epochs=120, learning_rate=0.08, l2=0.001):
    vectors = []
    targets = []
    domains = []
    for example in examples:
        if example.get("missing_image"):
            continue
        vector = feature_vector(
            example["prompt"],
            example["image_path"],
            label,
            grid_size=grid_size,
            prompt_dims=prompt_hash_dims,
            feature_version="v1",
            domain=example.get("domain"),
        )
        vectors.append(vector)
        targets.append(1.0 if label in set(example.get("target_cells", [])) else 0.0)
        domains.append(example.get("domain", ""))
    if not vectors:
        return {"weights": [], "bias": 0.0, "positive_count": 0, "negative_count": 0}
    width = len(vectors[0])
    weights = [0.0 for _ in range(width)]
    positives = sum(1 for target in targets if target)
    negatives = len(targets) - positives
    pos_weight = len(targets) / max(1.0, 2.0 * positives) if positives else 1.0
    neg_weight = len(targets) / max(1.0, 2.0 * negatives) if negatives else 1.0
    prior = (positives + 0.5) / (len(targets) + 1.0)
    bias = math_logit(prior)
    for _epoch in range(int(epochs)):
        grad_w = [0.0 for _ in range(width)]
        grad_b = 0.0
        for vector, target in zip(vectors, targets):
            score = sum(weights[index] * vector[index] for index in range(width)) + bias
            pred = sigmoid(score)
            scale = pos_weight if target else neg_weight
            error = (pred - target) * scale
            for index in range(width):
                grad_w[index] += error * vector[index]
            grad_b += error
        count = max(1, len(vectors))
        for index in range(width):
            grad = grad_w[index] / count + float(l2) * weights[index]
            weights[index] -= float(learning_rate) * grad
        bias -= float(learning_rate) * grad_b / count
    return {
        "weights": weights,
        "bias": bias,
        "positive_count": positives,
        "negative_count": negatives,
        "domains": sorted({str(domain) for domain in domains if domain}),
    }


def math_logit(value):
    import math

    value = max(1e-6, min(1.0 - 1e-6, float(value)))
    return math.log(value / (1.0 - value))


def build_model_v1(
    train_examples,
    grid_size=4,
    prompt_hash_dims=16,
    max_selected_cells=6,
    epochs=120,
    learning_rate=0.08,
    l2=0.001,
):
    labels = cell_labels(grid_size)
    cells = {
        label: train_logistic_cell(
            train_examples,
            label,
            grid_size,
            prompt_hash_dims,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        for label in labels
    }
    return {
        "schema_version": "ascr.grid_localizer_v1",
        "feature_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "grid_size": int(grid_size),
        "prompt_hash_dims": int(prompt_hash_dims),
        "max_selected_cells": int(max_selected_cells),
        "threshold": 0.0,
        "cells": cells,
        "training_summary": {
            "train_examples": len(train_examples),
            "missing_images": sum(1 for example in train_examples if example.get("missing_image")),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
        },
    }


def tune_threshold(model, train_examples):
    candidate_scores = []
    from ascr.training.localizer_model import score_cell

    labels = cell_labels(int(model["grid_size"]))
    for example in train_examples:
        if example.get("missing_image"):
            continue
        for label in labels:
            candidate_scores.append(score_cell(model, example["prompt"], example["image_path"], label, domain=example.get("domain")))
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


def train_grid_localizer(
    task,
    dataset,
    image_root,
    output_dir,
    eval_mode="holdout",
    train_ratio=0.8,
    seed=0,
    grid_size=4,
    prompt_hash_dims=None,
    max_selected_cells=6,
    epochs=120,
    learning_rate=0.08,
    l2=0.001,
):
    project_root = Path.cwd()
    examples = list(iter_localization_examples(dataset, image_root=image_root, project_root=project_root, grid_size=grid_size))
    if not examples:
        raise ValueError(f"No localization examples found in dataset: {dataset}")
    train_indices, eval_indices = split_examples(examples, eval_mode=eval_mode, train_ratio=train_ratio, seed=seed)
    train_examples = [examples[index] for index in train_indices]
    eval_examples = [examples[index] for index in eval_indices]
    if task == "grid-localizer-v1":
        dims = int(prompt_hash_dims if prompt_hash_dims is not None else 16)
        model = build_model_v1(
            train_examples,
            grid_size=grid_size,
            prompt_hash_dims=dims,
            max_selected_cells=max_selected_cells,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
    else:
        dims = int(prompt_hash_dims if prompt_hash_dims is not None else 8)
        model = build_model(
            train_examples,
            grid_size=grid_size,
            prompt_hash_dims=dims,
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
        "schema_version": f"ascr.{task}.metrics",
        "dataset": str(dataset),
        "image_root": str(image_root) if image_root else None,
        "row_count": len(examples),
        "train_rows": len(train_examples),
        "eval_rows": len(eval_examples),
        "eval_mode": eval_mode,
        "task": task,
        "train": train_metrics,
        "eval": eval_metrics,
        "missing_images": sum(1 for example in examples if example.get("missing_image")),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", eval_predictions)
    split_manifest = {
        "schema_version": f"ascr.{task}.split",
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
    parser.add_argument("--task", choices=["grid-localizer-v0", "grid-localizer-v1"], default="grid-localizer-v0")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="outputs/stage2_students/grid_localizer_v0")
    parser.add_argument("--eval-mode", choices=["resubstitution", "holdout"], default="holdout")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--prompt-hash-dims", type=int, default=None)
    parser.add_argument("--max-selected-cells", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = train_grid_localizer(
        args.task,
        args.dataset,
        args.image_root,
        args.output_dir,
        eval_mode=args.eval_mode,
        train_ratio=args.train_ratio,
        seed=args.seed,
        grid_size=args.grid_size,
        prompt_hash_dims=args.prompt_hash_dims,
        max_selected_cells=args.max_selected_cells,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

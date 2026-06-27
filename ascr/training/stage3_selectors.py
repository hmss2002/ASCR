"""Stage-3 selector baselines for self-corrupted token-repair datasets."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from ascr.analysis.token_locality import diff_energy_grid_from_paths
from ascr.core.schemas import GridCell
from ascr.corruption.vq_corruptor import token_indices_to_cell_labels
from ascr.training.localizer_model import cell_labels, feature_vector


DEFAULT_BASELINES = [
    "random",
    "token_prior",
    "rgb_diff_oracle",
    "rgb_localizer",
    "prompt_rgb_localizer",
]


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    return str(path)


def resolve_path(path, project_root=None):
    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    if project_root:
        candidates.append(Path(project_root) / raw)
    candidates.append(raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def target_cells(row, grid_size):
    grid_size = int(grid_size)
    key = f"coarse_labels_{grid_size}x{grid_size}"
    if row.get(key):
        return sorted({GridCell.from_any(label, grid_size).to_label() for label in row[key]})
    token_grid_size = int(row.get("token_grid_size") or 64)
    return token_indices_to_cell_labels(row.get("corruption_indices", []), token_grid_size, grid_size)


def selector_examples(dataset_path, grid_size, project_root=None):
    examples = []
    for idx, row in enumerate(read_jsonl(dataset_path)):
        clean_image = resolve_path(row.get("clean_image"), project_root=project_root)
        corrupted_image = resolve_path(row.get("corrupted_image"), project_root=project_root)
        examples.append({
            "idx": idx,
            "sample_id": row.get("sample_id", f"row{idx:04d}"),
            "prompt": row.get("prompt", ""),
            "corruption_type": row.get("corruption_type", "unknown"),
            "target_cells": target_cells(row, grid_size),
            "clean_image": str(clean_image),
            "corrupted_image": str(corrupted_image),
            "missing_clean_image": not clean_image.exists(),
            "missing_corrupted_image": not corrupted_image.exists(),
            "source": row,
        })
    return examples


def split_examples(examples, eval_mode="holdout", train_ratio=0.8, seed=0):
    indices = list(range(len(examples)))
    if eval_mode == "resubstitution":
        return indices, indices
    if eval_mode != "holdout":
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    if len(indices) <= 1:
        return indices, indices
    rng = random.Random(int(seed))
    by_type = {}
    for index, example in enumerate(examples):
        by_type.setdefault(example.get("corruption_type", "unknown"), []).append(index)

    train_indices = []
    eval_indices = []
    for group in by_type.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.extend(shuffled)
            continue
        train_count = int(len(shuffled) * float(train_ratio))
        train_count = max(1, min(len(shuffled) - 1, train_count))
        train_indices.extend(shuffled[:train_count])
        eval_indices.extend(shuffled[train_count:])
    train_indices = sorted(train_indices)
    eval_indices = sorted(eval_indices or train_indices)
    return train_indices, eval_indices


def _cell_to_rc(label, grid_size):
    cell = GridCell.from_any(label, int(grid_size))
    return cell.row, cell.col


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _prediction_distance(predicted, target, grid_size):
    if not predicted or not target:
        return None
    pred_rc = [_cell_to_rc(label, grid_size) for label in predicted]
    distances = []
    for label in target:
        row, col = _cell_to_rc(label, grid_size)
        distances.append(min(math.sqrt((row - prow) ** 2 + (col - pcol) ** 2) for prow, pcol in pred_rc))
    return sum(distances) / len(distances)


def evaluate_predictions(examples, prediction_map, grid_size, baseline, top_k):
    predictions = []
    for example in examples:
        predicted = list(prediction_map.get(example["sample_id"], []))[: int(top_k)]
        target = sorted(set(example.get("target_cells", [])))
        predicted_set = set(predicted)
        target_set = set(target)
        intersection = target_set & predicted_set
        union = target_set | predicted_set
        precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
        recall = len(intersection) / len(target_set) if target_set else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = len(intersection) / len(union) if union else 1.0
        predictions.append({
            "baseline": baseline,
            "grid_size": int(grid_size),
            "sample_id": example["sample_id"],
            "prompt": example.get("prompt", ""),
            "corruption_type": example.get("corruption_type", "unknown"),
            "target_cells": target,
            "predicted_cells": predicted,
            "selected_count": len(predicted),
            "precision_at_k": precision,
            "recall_at_k": recall,
            "f1_at_k": f1,
            "iou": iou,
            "hit_any": bool(intersection) if target_set else None,
            "exact_match": target_set == predicted_set,
            "mean_distance_to_target_cells": _prediction_distance(predicted, target, grid_size),
        })
    evaluated = [row for row in predictions if row["hit_any"] is not None]
    hits = sum(1 for row in evaluated if row["hit_any"])
    exact = sum(1 for row in evaluated if row["exact_match"])
    return {
        "schema_version": "ascr.stage3.selector_metrics.v1",
        "baseline": baseline,
        "grid_size": int(grid_size),
        "top_k": int(top_k),
        "row_count": len(predictions),
        "evaluated_rows": len(evaluated),
        "hit_any": hits,
        "hit_any_rate": hits / len(evaluated) if evaluated else None,
        "exact_match": exact,
        "exact_match_rate": exact / len(evaluated) if evaluated else None,
        "mean_precision_at_k": _mean(row["precision_at_k"] for row in evaluated),
        "mean_recall_at_k": _mean(row["recall_at_k"] for row in evaluated),
        "mean_f1_at_k": _mean(row["f1_at_k"] for row in evaluated),
        "mean_iou": _mean(row["iou"] for row in evaluated),
        "mean_distance_to_target_cells": _mean(row["mean_distance_to_target_cells"] for row in evaluated),
        "mean_selected_count": _mean(row["selected_count"] for row in predictions),
        "by_corruption_type": _metrics_by_corruption(predictions),
    }, predictions


def _metrics_by_corruption(predictions):
    grouped = {}
    for row in predictions:
        grouped.setdefault(row["corruption_type"], []).append(row)
    result = {}
    for kind, rows in sorted(grouped.items()):
        evaluated = [row for row in rows if row["hit_any"] is not None]
        result[kind] = {
            "row_count": len(rows),
            "evaluated_rows": len(evaluated),
            "hit_any_rate": sum(1 for row in evaluated if row["hit_any"]) / len(evaluated) if evaluated else None,
            "mean_f1_at_k": _mean(row["f1_at_k"] for row in evaluated),
            "mean_iou": _mean(row["iou"] for row in evaluated),
        }
    return result


def random_prediction_map(examples, grid_size, top_k, seed=0):
    labels = cell_labels(grid_size)
    predictions = {}
    for example in examples:
        digest = hashlib.md5(f"{seed}:{example['sample_id']}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:12], 16))
        shuffled = list(labels)
        rng.shuffle(shuffled)
        predictions[example["sample_id"]] = shuffled[: int(top_k)]
    return predictions


def train_token_prior(train_examples, grid_size, top_k):
    counts = Counter()
    for example in train_examples:
        counts.update(example.get("target_cells", []))
    labels = cell_labels(grid_size)
    ordered = sorted(labels, key=lambda label: (-counts.get(label, 0), label))
    model = {
        "schema_version": "ascr.stage3.token_prior_selector.v1",
        "created_at_utc": created_at_utc(),
        "grid_size": int(grid_size),
        "top_k": int(top_k),
        "cell_counts": {label: int(counts.get(label, 0)) for label in labels},
        "ordered_cells": ordered,
        "train_rows": len(train_examples),
    }
    return model


def token_prior_prediction_map(model, examples):
    selected = model["ordered_cells"][: int(model["top_k"])]
    return {example["sample_id"]: selected for example in examples}


def rgb_diff_oracle_prediction_map(examples, grid_size, top_k):
    predictions = {}
    for example in examples:
        if example.get("missing_clean_image") or example.get("missing_corrupted_image"):
            predictions[example["sample_id"]] = []
            continue
        energy = diff_energy_grid_from_paths(example["clean_image"], example["corrupted_image"], grid_size=grid_size)
        scored = []
        for row, values in enumerate(energy):
            for col, value in enumerate(values):
                scored.append((float(value), GridCell(row, col).to_label()))
        scored.sort(key=lambda item: (-item[0], item[1]))
        predictions[example["sample_id"]] = [label for _value, label in scored[: int(top_k)]]
    return predictions


def sigmoid(value):
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _feature(example, label, grid_size, feature_mode, prompt_hash_dims):
    prompt = example.get("prompt", "") if feature_mode == "prompt_rgb" else ""
    dims = int(prompt_hash_dims) if feature_mode == "prompt_rgb" else 1
    return feature_vector(
        prompt,
        example["corrupted_image"],
        label,
        grid_size=grid_size,
        prompt_dims=dims,
        feature_version="v1",
        domain=example.get("corruption_type", ""),
    )


def train_logistic_cell(train_examples, label, grid_size, feature_mode, prompt_hash_dims, epochs=120, learning_rate=0.08, l2=0.001):
    vectors = []
    targets = []
    for example in train_examples:
        if example.get("missing_corrupted_image"):
            continue
        vectors.append(_feature(example, label, grid_size, feature_mode, prompt_hash_dims))
        targets.append(1.0 if label in set(example.get("target_cells", [])) else 0.0)
    if not vectors:
        return {"weights": [], "bias": 0.0, "positive_count": 0, "negative_count": 0}
    width = len(vectors[0])
    weights = [0.0 for _ in range(width)]
    positives = sum(1 for target in targets if target)
    negatives = len(targets) - positives
    pos_weight = len(targets) / max(1.0, 2.0 * positives) if positives else 1.0
    neg_weight = len(targets) / max(1.0, 2.0 * negatives) if negatives else 1.0
    prior = (positives + 0.5) / (len(targets) + 1.0)
    bias = math.log(prior / (1.0 - prior))
    for _epoch in range(int(epochs)):
        grad_w = [0.0 for _ in range(width)]
        grad_b = 0.0
        for vector, target in zip(vectors, targets):
            pred = sigmoid(sum(weight * value for weight, value in zip(weights, vector)) + bias)
            scale = pos_weight if target else neg_weight
            error = (pred - target) * scale
            for index, value in enumerate(vector):
                grad_w[index] += error * value
            grad_b += error
        count = max(1, len(vectors))
        for index in range(width):
            grad = grad_w[index] / count + float(l2) * weights[index]
            weights[index] -= float(learning_rate) * grad
        bias -= float(learning_rate) * grad_b / count
    return {
        "weights": weights,
        "bias": bias,
        "positive_count": int(positives),
        "negative_count": int(negatives),
    }


def train_rgb_localizer(train_examples, grid_size, top_k, feature_mode, prompt_hash_dims=16, epochs=120, learning_rate=0.08, l2=0.001):
    labels = cell_labels(grid_size)
    cells = {
        label: train_logistic_cell(
            train_examples,
            label,
            grid_size,
            feature_mode,
            prompt_hash_dims,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        for label in labels
    }
    return {
        "schema_version": "ascr.stage3.rgb_localizer_selector.v1",
        "created_at_utc": created_at_utc(),
        "grid_size": int(grid_size),
        "top_k": int(top_k),
        "feature_mode": feature_mode,
        "prompt_hash_dims": int(prompt_hash_dims),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "l2": float(l2),
        "train_rows": len(train_examples),
        "cells": cells,
    }


def localizer_prediction_map(model, examples):
    grid_size = int(model["grid_size"])
    top_k = int(model["top_k"])
    feature_mode = model.get("feature_mode", "rgb")
    prompt_hash_dims = int(model.get("prompt_hash_dims", 16))
    predictions = {}
    for example in examples:
        if example.get("missing_corrupted_image"):
            predictions[example["sample_id"]] = []
            continue
        scored = []
        for label, cell_model in model["cells"].items():
            vector = _feature(example, label, grid_size, feature_mode, prompt_hash_dims)
            weights = cell_model.get("weights") or [0.0 for _ in vector]
            score = sum(weight * value for weight, value in zip(weights, vector)) + float(cell_model.get("bias", 0.0))
            scored.append((score, label))
        scored.sort(key=lambda item: (-item[0], item[1]))
        predictions[example["sample_id"]] = [label for _score, label in scored[:top_k]]
    return predictions


def run_baseline(
    baseline,
    train_examples,
    eval_examples,
    grid_size,
    top_k,
    output_dir,
    seed=0,
    prompt_hash_dims=16,
    epochs=120,
    learning_rate=0.08,
    l2=0.001,
):
    output_dir = Path(output_dir)
    baseline = str(baseline)
    model = None
    if baseline == "random":
        prediction_map = random_prediction_map(eval_examples, grid_size, top_k, seed=seed)
        model = {"schema_version": "ascr.stage3.random_selector.v1", "grid_size": int(grid_size), "top_k": int(top_k), "seed": int(seed)}
    elif baseline == "token_prior":
        model = train_token_prior(train_examples, grid_size, top_k)
        prediction_map = token_prior_prediction_map(model, eval_examples)
    elif baseline == "rgb_diff_oracle":
        prediction_map = rgb_diff_oracle_prediction_map(eval_examples, grid_size, top_k)
        model = {"schema_version": "ascr.stage3.rgb_diff_oracle.v1", "grid_size": int(grid_size), "top_k": int(top_k)}
    elif baseline == "rgb_localizer":
        model = train_rgb_localizer(
            train_examples,
            grid_size,
            top_k,
            "rgb",
            prompt_hash_dims=prompt_hash_dims,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        prediction_map = localizer_prediction_map(model, eval_examples)
    elif baseline == "prompt_rgb_localizer":
        model = train_rgb_localizer(
            train_examples,
            grid_size,
            top_k,
            "prompt_rgb",
            prompt_hash_dims=prompt_hash_dims,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        prediction_map = localizer_prediction_map(model, eval_examples)
    else:
        raise ValueError(f"Unsupported Stage-3 selector baseline: {baseline}")

    metrics, predictions = evaluate_predictions(eval_examples, prediction_map, grid_size, baseline, top_k)
    write_json(output_dir / "selector_model.json", model)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    return {
        "baseline": baseline,
        "grid_size": int(grid_size),
        "output_dir": str(output_dir),
        "metrics": metrics,
    }


def train_selector_suite(
    dataset,
    output_dir,
    grid_sizes=(4, 8, 16),
    baselines=None,
    eval_mode="holdout",
    train_ratio=0.75,
    seed=0,
    top_k=None,
    project_root=None,
    prompt_hash_dims=16,
    epochs=120,
    learning_rate=0.08,
    l2=0.001,
):
    baselines = list(baselines or DEFAULT_BASELINES)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    split_manifests = []
    for grid_size in [int(value) for value in grid_sizes]:
        examples = selector_examples(dataset, grid_size=grid_size, project_root=project_root)
        train_indices, eval_indices = split_examples(examples, eval_mode=eval_mode, train_ratio=train_ratio, seed=seed)
        train_examples = [examples[index] for index in train_indices]
        eval_examples = [examples[index] for index in eval_indices]
        k = int(top_k if top_k is not None else max(1, round(len(cell_labels(grid_size)) ** 0.5 / 2.0)))
        split_manifest = {
            "schema_version": "ascr.stage3.selector_split.v1",
            "dataset": str(dataset),
            "grid_size": grid_size,
            "seed": int(seed),
            "eval_mode": eval_mode,
            "train_ratio": float(train_ratio),
            "row_count": len(examples),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "train_sample_ids": [examples[index]["sample_id"] for index in train_indices],
            "eval_sample_ids": [examples[index]["sample_id"] for index in eval_indices],
            "missing_clean_images": sum(1 for example in examples if example.get("missing_clean_image")),
            "missing_corrupted_images": sum(1 for example in examples if example.get("missing_corrupted_image")),
        }
        split_path = output_dir / f"grid{grid_size}" / "split_manifest.json"
        write_json(split_path, split_manifest)
        split_manifests.append(str(split_path))
        for baseline in baselines:
            result = run_baseline(
                baseline,
                train_examples,
                eval_examples,
                grid_size,
                k,
                output_dir / f"grid{grid_size}" / baseline,
                seed=seed,
                prompt_hash_dims=prompt_hash_dims,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
            )
            results.append(result)
    summary = {
        "schema_version": "ascr.stage3.selector_suite_summary.v1",
        "created_at_utc": created_at_utc(),
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "grid_sizes": [int(value) for value in grid_sizes],
        "baselines": baselines,
        "eval_mode": eval_mode,
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "top_k": top_k,
        "split_manifests": split_manifests,
        "results": [
            {
                "baseline": result["baseline"],
                "grid_size": result["grid_size"],
                "output_dir": result["output_dir"],
                "hit_any_rate": result["metrics"].get("hit_any_rate"),
                "mean_f1_at_k": result["metrics"].get("mean_f1_at_k"),
                "mean_iou": result["metrics"].get("mean_iou"),
                "mean_distance_to_target_cells": result["metrics"].get("mean_distance_to_target_cells"),
            }
            for result in results
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary

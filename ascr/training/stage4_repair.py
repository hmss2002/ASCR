"""Phase-4 hidden-state repair-head scaffolding for Stage-3 self-corruption."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from ascr.core.schemas import GridCell
from ascr.generators.lumina_native import LuminaNativeEngine
from ascr.training.stage3_selectors import read_jsonl, target_cells, write_json, write_jsonl


FEATURE_ROW_SCHEMA = "ascr.stage4.hidden_features.row.v1"
REPAIR_HEAD_SCHEMA = "ascr.stage4.repair_head.v1"


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_labels(grid_size):
    return [GridCell(row, col).to_label() for row in range(int(grid_size)) for col in range(int(grid_size))]


def split_rows(rows, eval_mode="holdout", train_ratio=0.8, seed=0):
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


def token_positions_for_cell(cell_label, grid_size, token_grid_size, code_start, include_newlines=True):
    cell = GridCell.from_any(cell_label, int(grid_size))
    factor = int(token_grid_size) // int(grid_size)
    positions = []
    row0 = cell.row * factor
    col0 = cell.col * factor
    stride = int(token_grid_size) + (1 if include_newlines else 0)
    for row in range(row0, row0 + factor):
        for col in range(col0, col0 + factor):
            positions.append(int(code_start) + row * stride + col)
    return positions


def _project_vector(torch, vector, feature_dim, seed):
    if feature_dim is None or int(feature_dim) <= 0 or int(feature_dim) >= int(vector.numel()):
        return vector.float().detach().cpu().tolist()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        int(vector.numel()),
        int(feature_dim),
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(float(feature_dim))
    projected = vector.float().detach().cpu().matmul(matrix)
    return projected.tolist()


def _round_feature(values, digits=6):
    return [round(float(value), int(digits)) for value in values]


def _forward_hidden_states(engine, prompt, corrupted_vq_ids):
    engine._load()
    prompt_ids, _uncon_ids, code_start = engine._build_prompt_ids(prompt, list(corrupted_vq_ids))
    model = engine._model
    torch = engine._torch
    with torch.no_grad():
        try:
            outputs = model(input_ids=prompt_ids, output_hidden_states=True, return_dict=True)
        except TypeError:
            outputs = model(prompt_ids, output_hidden_states=True, return_dict=True)
    hidden_states = None
    if hasattr(outputs, "hidden_states"):
        hidden_states = outputs.hidden_states
    elif isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    elif isinstance(outputs, (list, tuple)):
        hidden_states = next((item for item in outputs if isinstance(item, (list, tuple))), None)
    if not hidden_states:
        raise RuntimeError("Lumina forward did not return hidden_states")
    return hidden_states, int(code_start), tuple(prompt_ids.shape)


def probe_hidden_state_support(dataset, output_dir, limit=1, **engine_kwargs):
    rows = read_jsonl(dataset)
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = LuminaNativeEngine(**engine_kwargs)
    probe_rows = []
    failures = []
    for row in rows:
        try:
            corrupted_vq_ids = read_json(row["corrupted_vq_ids_path"])
            hidden_states, code_start, input_shape = _forward_hidden_states(engine, row["prompt"], corrupted_vq_ids)
            shapes = [list(state.shape) for state in hidden_states]
            probe_rows.append({
                "sample_id": row.get("sample_id"),
                "prompt": row.get("prompt"),
                "corruption_type": row.get("corruption_type"),
                "input_shape": list(input_shape),
                "code_start": code_start,
                "hidden_state_count": len(hidden_states),
                "hidden_state_shapes": shapes,
                "last_hidden_shape": shapes[-1] if shapes else None,
                "status": "ok",
            })
        except Exception as exc:
            failures.append({
                "sample_id": row.get("sample_id"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    report = {
        "schema_version": "ascr.stage4.hidden_state_probe.v1",
        "created_at_utc": created_at_utc(),
        "dataset": str(dataset),
        "row_count": len(rows),
        "ok_count": len(probe_rows),
        "failure_count": len(failures),
        "supports_hidden_states": bool(probe_rows and not failures),
        "rows": probe_rows,
        "failures": failures,
    }
    write_json(output_dir / "hidden_state_probe.json", report)
    return report


def extract_hidden_features(
    dataset,
    output_dir,
    grid_size=16,
    hidden_layer=-1,
    feature_dim=128,
    projection_seed=0,
    limit=None,
    **engine_kwargs,
):
    rows = read_jsonl(dataset)
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = LuminaNativeEngine(**engine_kwargs)
    feature_rows = []
    failures = []
    labels = cell_labels(grid_size)
    for row in rows:
        try:
            corrupted_vq_ids = read_json(row["corrupted_vq_ids_path"])
            hidden_states, code_start, _input_shape = _forward_hidden_states(engine, row["prompt"], corrupted_vq_ids)
            torch = engine._torch
            selected = hidden_states[int(hidden_layer)][0]
            target = set(target_cells(row, grid_size))
            cells = []
            for label in labels:
                positions = token_positions_for_cell(
                    label,
                    grid_size,
                    int(row.get("token_grid_size") or engine.token_grid_size),
                    code_start,
                    include_newlines=True,
                )
                positions = [pos for pos in positions if 0 <= pos < selected.shape[0]]
                if not positions:
                    continue
                index_t = torch.tensor(positions, device=selected.device, dtype=torch.long)
                vector = selected.index_select(0, index_t).mean(dim=0)
                cells.append({
                    "label": label,
                    "target": label in target,
                    "feature": _round_feature(_project_vector(torch, vector, feature_dim, projection_seed)),
                })
            feature_rows.append({
                "schema_version": FEATURE_ROW_SCHEMA,
                "sample_id": row.get("sample_id"),
                "prompt": row.get("prompt"),
                "corruption_type": row.get("corruption_type"),
                "grid_size": int(grid_size),
                "target_cells": sorted(target),
                "hidden_layer": int(hidden_layer),
                "feature_dim": int(feature_dim) if feature_dim else None,
                "projection_seed": int(projection_seed),
                "cells": cells,
            })
        except Exception as exc:
            failures.append({
                "sample_id": row.get("sample_id"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    features_path = output_dir / "hidden_features.jsonl"
    write_jsonl(features_path, feature_rows)
    manifest = {
        "schema_version": "ascr.stage4.hidden_features_manifest.v1",
        "created_at_utc": created_at_utc(),
        "dataset": str(dataset),
        "features": str(features_path),
        "row_count": len(feature_rows),
        "failure_count": len(failures),
        "grid_size": int(grid_size),
        "hidden_layer": int(hidden_layer),
        "feature_dim": int(feature_dim) if feature_dim else None,
        "projection_seed": int(projection_seed),
        "failures": failures,
    }
    write_json(output_dir / "hidden_features_manifest.json", manifest)
    return manifest


def sigmoid(value):
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def train_binary_logistic(vectors, targets, epochs=120, learning_rate=0.08, l2=0.001):
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
            error = (pred - float(target)) * scale
            for index, value in enumerate(vector):
                grad_w[index] += error * float(value)
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


def _score(cell_model, vector):
    return sum(float(w) * float(v) for w, v in zip(cell_model.get("weights", []), vector)) + float(cell_model.get("bias", 0.0))


def _metrics(rows, predictions):
    evaluated = []
    for row, predicted in zip(rows, predictions):
        target = set(row.get("target_cells", []))
        pred = set(predicted["predicted_cells"])
        inter = target & pred
        union = target | pred
        precision = len(inter) / len(pred) if pred else 0.0
        recall = len(inter) / len(target) if target else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = len(inter) / len(union) if union else 1.0
        predicted["precision_at_k"] = precision
        predicted["recall_at_k"] = recall
        predicted["f1_at_k"] = f1
        predicted["iou"] = iou
        predicted["hit_any"] = bool(inter) if target else None
        evaluated.append(predicted)
    valid = [row for row in evaluated if row["hit_any"] is not None]
    return {
        "evaluated_rows": len(valid),
        "hit_any": sum(1 for row in valid if row["hit_any"]),
        "hit_any_rate": sum(1 for row in valid if row["hit_any"]) / len(valid) if valid else None,
        "mean_precision_at_k": _mean(row["precision_at_k"] for row in valid),
        "mean_recall_at_k": _mean(row["recall_at_k"] for row in valid),
        "mean_f1_at_k": _mean(row["f1_at_k"] for row in valid),
        "mean_iou": _mean(row["iou"] for row in valid),
    }


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def train_repair_head(
    features_jsonl,
    output_dir,
    eval_mode="holdout",
    train_ratio=0.8,
    seed=0,
    top_k=4,
    epochs=120,
    learning_rate=0.08,
    l2=0.001,
):
    rows = read_jsonl(features_jsonl)
    if not rows:
        raise ValueError(f"No feature rows found in {features_jsonl}")
    train_indices, eval_indices = split_rows(rows, eval_mode=eval_mode, train_ratio=train_ratio, seed=seed)
    train_rows = [rows[index] for index in train_indices]
    eval_rows = [rows[index] for index in eval_indices]
    labels = sorted({cell["label"] for row in rows for cell in row.get("cells", [])})
    cells = {}
    for label in labels:
        vectors = []
        targets = []
        for row in train_rows:
            for cell in row.get("cells", []):
                if cell.get("label") == label:
                    vectors.append(cell["feature"])
                    targets.append(bool(cell.get("target")))
                    break
        cells[label] = train_binary_logistic(
            vectors,
            targets,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
    model = {
        "schema_version": REPAIR_HEAD_SCHEMA,
        "created_at_utc": created_at_utc(),
        "features": str(features_jsonl),
        "grid_size": int(rows[0].get("grid_size")),
        "hidden_layer": int(rows[0].get("hidden_layer")),
        "feature_dim": rows[0].get("feature_dim"),
        "top_k": int(top_k),
        "cells": cells,
        "training_summary": {
            "row_count": len(rows),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "eval_mode": eval_mode,
            "train_ratio": float(train_ratio),
            "seed": int(seed),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
        },
    }
    predictions = []
    for row in eval_rows:
        scored = []
        for cell in row.get("cells", []):
            label = cell["label"]
            scored.append((_score(cells[label], cell["feature"]), label))
        scored.sort(key=lambda item: (-item[0], item[1]))
        predictions.append({
            "sample_id": row.get("sample_id"),
            "target_cells": row.get("target_cells", []),
            "predicted_cells": [label for _score_value, label in scored[: int(top_k)]],
            "top_scores": [{"cell": label, "score": score_value} for score_value, label in scored[: max(8, int(top_k))]],
        })
    metrics = {
        "schema_version": "ascr.stage4.repair_head_metrics.v1",
        "features": str(features_jsonl),
        "row_count": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "top_k": int(top_k),
        "eval": _metrics(eval_rows, predictions),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "repair_head.json", model)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "split_manifest.json", {
        "schema_version": "ascr.stage4.repair_head_split.v1",
        "features": str(features_jsonl),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_sample_ids": [rows[index].get("sample_id") for index in train_indices],
        "eval_sample_ids": [rows[index].get("sample_id") for index in eval_indices],
    })
    return {"model": str(output_dir / "repair_head.json"), "metrics": metrics, "output_dir": str(output_dir)}

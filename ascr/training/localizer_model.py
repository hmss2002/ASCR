import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache

from ascr.core.schemas import GridCell


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def cell_labels(grid_size):
    return [GridCell(row, col).to_label() for row in range(grid_size) for col in range(grid_size)]


def prompt_hash_features(prompt, dims=8):
    values = [0.0 for _ in range(int(dims))]
    tokens = [token.strip().lower() for token in str(prompt or "").replace(",", " ").split() if token.strip()]
    if not tokens:
        return values
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = digest[0] % len(values)
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        values[bucket] += sign
    scale = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / scale for value in values]


def _read_ppm_rgb(path):
    data = Path(path).read_bytes()
    if not data.startswith(b"P3"):
        raise ValueError("only ASCII P3 PPM fallback is supported without Pillow")
    tokens = []
    for line in data.decode("ascii", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "P3":
            continue
        tokens.extend(line.split())
    width = int(tokens[0])
    height = int(tokens[1])
    max_value = max(1, int(tokens[2]))
    raw = [int(value) / max_value for value in tokens[3:]]
    pixels = []
    offset = 0
    for _row in range(height):
        line = []
        for _col in range(width):
            line.append((raw[offset], raw[offset + 1], raw[offset + 2]))
            offset += 3
        pixels.append(line)
    return pixels, width, height


def load_rgb_pixels(path):
    path = Path(path)
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            flat = list(image.getdata())
            pixels = []
            offset = 0
            for _row in range(height):
                row = []
                for _col in range(width):
                    r, g, b = flat[offset]
                    row.append((r / 255.0, g / 255.0, b / 255.0))
                    offset += 1
                pixels.append(row)
            return pixels, width, height
    except Exception:
        return _read_ppm_rgb(path)


@lru_cache(maxsize=512)
def _cell_rgb_means_cached(image_path, grid_size=4):
    pixels, width, height = load_rgb_pixels(image_path)
    grid_size = int(grid_size)
    means = {}
    global_sum = [0.0, 0.0, 0.0]
    global_count = 0
    for row in range(height):
        for col in range(width):
            r, g, b = pixels[row][col]
            global_sum[0] += r
            global_sum[1] += g
            global_sum[2] += b
            global_count += 1
    global_mean = [value / max(1, global_count) for value in global_sum]
    for grid_row in range(grid_size):
        for grid_col in range(grid_size):
            r0 = int(grid_row * height / grid_size)
            r1 = int((grid_row + 1) * height / grid_size)
            c0 = int(grid_col * width / grid_size)
            c1 = int((grid_col + 1) * width / grid_size)
            accum = [0.0, 0.0, 0.0]
            count = 0
            for row in range(r0, max(r0 + 1, r1)):
                for col in range(c0, max(c0 + 1, c1)):
                    r, g, b = pixels[min(row, height - 1)][min(col, width - 1)]
                    accum[0] += r
                    accum[1] += g
                    accum[2] += b
                    count += 1
            label = GridCell(grid_row, grid_col).to_label()
            means[label] = [value / max(1, count) for value in accum] + global_mean
    return means


def cell_rgb_means(image_path, grid_size=4):
    return dict(_cell_rgb_means_cached(str(Path(image_path).resolve()), int(grid_size)))


def feature_vector(prompt, image_path, cell_label, grid_size=4, prompt_dims=8):
    cell = GridCell.from_any(cell_label, grid_size)
    rgb = cell_rgb_means(image_path, grid_size=grid_size)[cell.to_label()]
    local = rgb[:3]
    global_mean = rgb[3:]
    diff = [local[index] - global_mean[index] for index in range(3)]
    coords = [
        cell.row / max(1, grid_size - 1),
        cell.col / max(1, grid_size - 1),
    ]
    return local + global_mean + diff + coords + prompt_hash_features(prompt, dims=prompt_dims)


def vector_mean(vectors):
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def resolve_training_image(path, image_root=None, project_root=None):
    if not path:
        return None
    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    if image_root:
        candidates.append(Path(image_root) / raw)
    if project_root:
        candidates.append(Path(project_root) / raw)
    candidates.append(raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def localization_cells(localization, grid_size=4):
    cells = []
    evaluation = localization.get("evaluation", {})
    for region in evaluation.get("regions", []) or []:
        for cell in region.get("cells", []) or []:
            label = cell.get("label") if isinstance(cell, dict) else str(cell)
            if label:
                cells.append(GridCell.from_any(label, grid_size).to_label())
    return sorted(set(cells))


def iter_localization_examples(dataset_path, image_root=None, project_root=None, grid_size=4):
    for row in read_jsonl(dataset_path):
        for localization in row.get("localizations", []) or []:
            image_path = resolve_training_image(localization.get("grid_image"), image_root=image_root, project_root=project_root)
            if not image_path or not Path(image_path).exists():
                yield {
                    "sample_id": localization.get("sample_id", row.get("sample_id")),
                    "prompt": localization.get("prompt", row.get("prompt", "")),
                    "image_path": str(image_path) if image_path else None,
                    "target_cells": localization_cells(localization, grid_size=grid_size),
                    "missing_image": True,
                }
                continue
            yield {
                "sample_id": localization.get("sample_id", row.get("sample_id")),
                "prompt": localization.get("prompt", row.get("prompt", "")),
                "image_path": str(image_path),
                "target_cells": localization_cells(localization, grid_size=grid_size),
                "missing_image": False,
            }


def score_cell(model, prompt, image_path, cell_label):
    vector = feature_vector(
        prompt,
        image_path,
        cell_label,
        grid_size=int(model.get("grid_size", 4)),
        prompt_dims=int(model.get("prompt_hash_dims", 8)),
    )
    cell_model = model["cells"].get(cell_label, {})
    positive = cell_model.get("positive_centroid") or model.get("global_positive_centroid") or [0.0 for _ in vector]
    negative = cell_model.get("negative_centroid") or model.get("global_negative_centroid") or [0.0 for _ in vector]
    return dot(vector, positive) - dot(vector, negative) + float(cell_model.get("bias", model.get("bias", 0.0)))


def predict_cells(model, prompt, image_path):
    labels = cell_labels(int(model.get("grid_size", 4)))
    scored = [(label, score_cell(model, prompt, image_path, label)) for label in labels]
    scored.sort(key=lambda item: (-item[1], item[0]))
    threshold = float(model.get("threshold", 0.0))
    max_selected = int(model.get("max_selected_cells", 6))
    selected = [label for label, score in scored if score >= threshold][:max_selected]
    return selected, scored


def evaluate_predictions(examples, model):
    predictions = []
    evaluated = 0
    hit_any = 0
    total_f1 = 0.0
    exact_match = 0
    for example in examples:
        if example.get("missing_image"):
            continue
        predicted, scored = predict_cells(model, example["prompt"], example["image_path"])
        target = sorted(set(example.get("target_cells", [])))
        predicted_set = set(predicted)
        target_set = set(target)
        if target_set or predicted_set:
            evaluated += 1
            if target_set & predicted_set:
                hit_any += 1
            if target_set == predicted_set:
                exact_match += 1
            precision = len(target_set & predicted_set) / len(predicted_set) if predicted_set else 0.0
            recall = len(target_set & predicted_set) / len(target_set) if target_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            total_f1 += f1
        predictions.append({
            "sample_id": example.get("sample_id"),
            "prompt": example.get("prompt"),
            "target_cells": target,
            "predicted_cells": predicted,
            "top_scores": [{"cell": label, "score": score} for label, score in scored[:8]],
        })
    return {
        "evaluated_rows": evaluated,
        "hit_any": hit_any,
        "hit_any_rate": hit_any / evaluated if evaluated else None,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / evaluated if evaluated else None,
        "mean_f1": total_f1 / evaluated if evaluated else None,
    }, predictions

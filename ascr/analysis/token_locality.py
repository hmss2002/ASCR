"""Token-to-image locality metrics for Stage-3 self-corruption probes."""

import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from ascr.corruption.vq_corruptor import token_indices_to_cell_labels
from ascr.training.localizer_model import load_rgb_pixels


def diff_energy_grid(clean_pixels, corrupted_pixels, grid_size: int):
    """Aggregate absolute RGB difference into a coarse grid."""
    height = len(clean_pixels)
    width = len(clean_pixels[0]) if height else 0
    if height != len(corrupted_pixels) or any(len(clean_pixels[row]) != len(corrupted_pixels[row]) for row in range(height)):
        raise ValueError("Clean and corrupted images must have the same dimensions")
    grid_size = int(grid_size)
    energy = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    for row in range(height):
        grid_row = min(grid_size - 1, int(row * grid_size / max(1, height)))
        for col in range(width):
            grid_col = min(grid_size - 1, int(col * grid_size / max(1, width)))
            clean = clean_pixels[row][col]
            corrupt = corrupted_pixels[row][col]
            energy[grid_row][grid_col] += sum(abs(float(a) - float(b)) for a, b in zip(clean, corrupt))
    return energy


def diff_energy_grid_from_paths(clean_image_path, corrupted_image_path, grid_size: int):
    clean_pixels, clean_width, clean_height = load_rgb_pixels(clean_image_path)
    corrupted_pixels, corrupted_width, corrupted_height = load_rgb_pixels(corrupted_image_path)
    if (clean_width, clean_height) != (corrupted_width, corrupted_height):
        raise ValueError("Clean and corrupted images must have the same dimensions")
    return diff_energy_grid(clean_pixels, corrupted_pixels, grid_size=grid_size)


def project_token_indices(indices: Iterable[Tuple[int, int]], token_grid_size: int, grid_size: int):
    token_grid_size = int(token_grid_size)
    grid_size = int(grid_size)
    if token_grid_size % grid_size != 0:
        raise ValueError("token_grid_size must be divisible by grid_size")
    factor = token_grid_size // grid_size
    return sorted({(int(row) // factor, int(col) // factor) for row, col in indices})


def _total_energy(energy_grid: Sequence[Sequence[float]]):
    return sum(sum(float(value) for value in row) for row in energy_grid)


def _weighted_centroid(energy_grid: Sequence[Sequence[float]]):
    total = _total_energy(energy_grid)
    if total <= 0:
        return None
    row_sum = 0.0
    col_sum = 0.0
    for row, values in enumerate(energy_grid):
        for col, value in enumerate(values):
            weight = float(value)
            row_sum += row * weight
            col_sum += col * weight
    return row_sum / total, col_sum / total


def _corruption_centroid(selected_indices: Iterable[Tuple[int, int]], token_grid_size: int, grid_size: int):
    selected = list(selected_indices)
    if not selected:
        return None
    scale = float(grid_size) / float(token_grid_size)
    row = sum((int(index[0]) + 0.5) * scale - 0.5 for index in selected) / len(selected)
    col = sum((int(index[1]) + 0.5) * scale - 0.5 for index in selected) / len(selected)
    return row, col


def _top_cells(energy_grid: Sequence[Sequence[float]], count: int):
    cells = []
    for row, values in enumerate(energy_grid):
        for col, value in enumerate(values):
            cells.append((float(value), row, col))
    cells.sort(reverse=True)
    return [(row, col) for _value, row, col in cells[: int(count)]]


def _energy_within_radius(energy_grid, selected_cells, radius):
    selected = set(selected_cells)
    total = 0.0
    for row, values in enumerate(energy_grid):
        for col, value in enumerate(values):
            if any(max(abs(row - srow), abs(col - scol)) <= radius for srow, scol in selected):
                total += float(value)
    return total


def summarise_locality(
    energy_grid: Sequence[Sequence[float]],
    selected_indices: Iterable[Tuple[int, int]],
    token_grid_size: int,
    energy_fraction: float = 0.8,
):
    """Summarise whether visual changes concentrate near selected token cells."""
    grid_size = len(energy_grid)
    selected_indices = list(selected_indices)
    selected_cells = project_token_indices(selected_indices, token_grid_size, grid_size)
    selected_cell_set = set(selected_cells)
    total = _total_energy(energy_grid)
    inside = sum(float(energy_grid[row][col]) for row, col in selected_cells)
    outside = max(0.0, total - inside)
    weighted = _weighted_centroid(energy_grid)
    corruption = _corruption_centroid(selected_indices, token_grid_size, grid_size)
    displacement = None
    if weighted is not None and corruption is not None:
        displacement = math.sqrt((weighted[0] - corruption[0]) ** 2 + (weighted[1] - corruption[1]) ** 2)
    top1 = _top_cells(energy_grid, 1)
    topk = _top_cells(energy_grid, max(1, len(selected_cells)))
    effective_radius = None
    if total > 0 and selected_cells:
        for radius in range(grid_size + 1):
            if _energy_within_radius(energy_grid, selected_cells, radius) / total >= float(energy_fraction):
                effective_radius = radius
                break
    return {
        "grid_size": grid_size,
        "token_grid_size": int(token_grid_size),
        "selected_cells": token_indices_to_cell_labels(selected_indices, token_grid_size, grid_size),
        "selected_cell_count": len(selected_cells),
        "total_energy": total,
        "inside_energy": inside,
        "outside_energy": outside,
        "inside_outside_energy_ratio": inside / outside if outside > 0 else None,
        "inside_energy_fraction": inside / total if total > 0 else 0.0,
        "center_displacement_cells": displacement,
        "top1_cell_hit": bool(top1 and top1[0] in selected_cell_set),
        "topk_cell_hit": any(cell in selected_cell_set for cell in topk),
        "effective_radius_cells": effective_radius,
        "energy_fraction_target": float(energy_fraction),
    }


def write_heatmap_ppm(energy_grid: Sequence[Sequence[float]], output_path, scale: int = 32):
    """Write a simple red heatmap as ASCII PPM without requiring Pillow."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid_size = len(energy_grid)
    max_value = max([float(value) for row in energy_grid for value in row] or [0.0])
    scale = max(1, int(scale))
    width = grid_size * scale
    height = grid_size * scale
    lines = ["P3", f"{width} {height}", "255"]
    for row in range(height):
        source_row = row // scale
        values = []
        for col in range(width):
            source_col = col // scale
            energy = float(energy_grid[source_row][source_col])
            intensity = int(round(255 * energy / max_value)) if max_value > 0 else 0
            values.extend([str(intensity), "0", str(255 - intensity)])
        lines.append(" ".join(values))
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return str(output)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)

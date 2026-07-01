"""Bridge Stage-4 MMU localization outputs to token reopening masks."""

from __future__ import annotations

import json
from pathlib import Path

from ascr.core.schemas import GridCell, TokenReopenMask
from ascr.distill.teacher import extract_json_object
from ascr.grids.projection import project_cells_to_token_mask
from ascr.training.stage4_mmu_lora import safe_parse_mmu_localization_payload


def _read_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _maybe_load(value):
    if isinstance(value, Path):
        try:
            if value.exists() and value.is_file():
                return _read_json_or_jsonl(value)
        except OSError:
            return value
    if isinstance(value, str):
        text = value.strip()
        if not text or len(text) >= 256 or text[0] in "{[" or "\n" in text:
            return value
        try:
            candidate = Path(text)
            if candidate.exists() and candidate.is_file():
                return _read_json_or_jsonl(candidate)
        except OSError:
            return value
    return value


def _flatten(values):
    if values is None:
        return []
    if isinstance(values, (str, dict)):
        return [values]
    if (
        isinstance(values, (list, tuple))
        and len(values) == 2
        and all(isinstance(item, (int, float)) or str(item).lstrip("-").isdigit() for item in values)
    ):
        return [values]
    if isinstance(values, (list, tuple, set)):
        out = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [values]


def _cells_from_semantic(evaluation):
    cells = []
    for region in evaluation.actionable_regions():
        cells.extend(region.cells)
    return cells


def _cells_from_dict(payload, grid_size, max_selected_cells):
    if payload.get("mask") and payload.get("selected_indices") is not None:
        return payload.get("selected_indices") or []
    for key in ("predicted_cells", "cells", "selected_cells", "grid_cells", "cell_labels"):
        if payload.get(key) is not None:
            return payload.get(key)
    for key in ("normalised_payload", "parsed_payload", "raw_payload"):
        if payload.get(key) is not None:
            return extract_cells(payload[key], grid_size=grid_size, max_selected_cells=max_selected_cells)
    parsed = payload.get("parsed")
    if isinstance(parsed, dict) and parsed.get("regions") is not None:
        evaluation, _normalised = safe_parse_mmu_localization_payload(
            parsed,
            grid_size=grid_size,
            max_selected_cells=max_selected_cells,
        )
        return _cells_from_semantic(evaluation)
    evaluation, _normalised = safe_parse_mmu_localization_payload(
        payload,
        grid_size=grid_size,
        max_selected_cells=max_selected_cells,
    )
    return _cells_from_semantic(evaluation)


def extract_cells(value, grid_size=4, max_selected_cells=16):
    """Extract grid cells from raw JSON, probe rows, probe JSONL, or cell lists."""

    value = _maybe_load(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = extract_json_object(value)
    if isinstance(value, dict):
        return [GridCell.from_any(cell, grid_size) for cell in _flatten(_cells_from_dict(value, grid_size, max_selected_cells))]
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            rows = [row for row in value if row.get("status") in {None, "parsed"}]
            if len(rows) == 1:
                return extract_cells(rows[0], grid_size=grid_size, max_selected_cells=max_selected_cells)
            merged = []
            for row in rows:
                merged.extend(extract_cells(row, grid_size=grid_size, max_selected_cells=max_selected_cells))
            return sorted({(cell.row, cell.col): cell for cell in merged}.values(), key=lambda cell: (cell.row, cell.col))
        return [GridCell.from_any(cell, grid_size) for cell in _flatten(value)]
    return []


class MMULocalizerSelector:
    """Convert MMU-localizer cell predictions to a ``TokenReopenMask``."""

    def __init__(
        self,
        probe_summary_or_cells,
        grid_size,
        token_grid_size=64,
        dilation=0,
        max_selected_cells=16,
    ):
        self.source = probe_summary_or_cells
        self.grid_size = int(grid_size)
        self.token_grid_size = int(token_grid_size)
        self.dilation = int(dilation)
        self.max_selected_cells = int(max_selected_cells)

    def cells(self):
        cells = extract_cells(
            self.source,
            grid_size=self.grid_size,
            max_selected_cells=self.max_selected_cells,
        )
        deduped = {(cell.row, cell.col): cell for cell in cells}
        selected = [deduped[key] for key in sorted(deduped)]
        return selected[: self.max_selected_cells]

    def to_token_mask(self):
        cells = self.cells()
        if not cells:
            return TokenReopenMask.empty(self.token_grid_size)
        return project_cells_to_token_mask(
            cells,
            coarse_grid_size=self.grid_size,
            token_grid_size=self.token_grid_size,
            dilation=self.dilation,
        )

    def stats(self):
        cells = self.cells()
        mask = self.to_token_mask()
        return {
            "grid_size": self.grid_size,
            "token_grid_size": self.token_grid_size,
            "dilation": self.dilation,
            "max_selected_cells": self.max_selected_cells,
            "cell_count": len(cells),
            "cells": [cell.to_label() for cell in cells],
            "selected_token_count": mask.count(),
        }

    def select(self, _evaluation=None):
        return self.to_token_mask()

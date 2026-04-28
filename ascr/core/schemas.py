from dataclasses import dataclass, field
import json
import re
from typing import Any, List, Optional


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int

    def validate(self, grid_size=4):
        if not (0 <= self.row < grid_size and 0 <= self.col < grid_size):
            raise SchemaError(f"Grid cell out of range: row={self.row}, col={self.col}, grid_size={grid_size}")
        return self

    @classmethod
    def from_any(cls, value, grid_size=4):
        if isinstance(value, GridCell):
            return value.validate(grid_size)
        if isinstance(value, dict):
            row = value.get("row", value.get("r"))
            col = value.get("col", value.get("c"))
            if row is None or col is None:
                label = value.get("cell", value.get("label"))
                if label is not None:
                    return cls.from_any(label, grid_size)
            return cls(int(row), int(col)).validate(grid_size)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(int(value[0]), int(value[1])).validate(grid_size)
        if isinstance(value, str):
            text = value.strip().upper()
            match = re.fullmatch(r"([A-D])([1-4])", text)
            if match:
                return cls(ord(match.group(1)) - ord("A"), int(match.group(2)) - 1).validate(grid_size)
            match = re.fullmatch(r"R(\d+)C(\d+)", text)
            if match:
                return cls(int(match.group(1)), int(match.group(2))).validate(grid_size)
            match = re.fullmatch(r"(\d+)\s*,\s*(\d+)", text)
            if match:
                return cls(int(match.group(1)), int(match.group(2))).validate(grid_size)
        raise SchemaError(f"Cannot parse grid cell from {value!r}")

    def to_label(self):
        label_char = chr(ord("A") + self.row)
        return f"{label_char}{self.col + 1}"

    def to_dict(self):
        return {"row": self.row, "col": self.col, "label": self.to_label()}


@dataclass
class RegionSelection:
    cells: List[GridCell]
    reason: str = ""
    confidence: float = 1.0
    error_type: str = "semantic"
    action: str = "reopen"

    @classmethod
    def from_any(cls, value, grid_size=4):
        if isinstance(value, RegionSelection):
            return value
        if not isinstance(value, dict):
            raise SchemaError("Region selection must be an object")
        raw_cells = value.get("cells", value.get("grid_cells", value.get("selected_cells", [])))
        if isinstance(raw_cells, (str, dict)):
            raw_cells = [raw_cells]
        cells = [GridCell.from_any(cell, grid_size) for cell in raw_cells]
        if not cells:
            raise SchemaError("Region selection has no valid cells")
        return cls(
            cells=cells,
            reason=str(value.get("reason", value.get("description", ""))),
            confidence=float(value.get("confidence", 1.0)),
            error_type=str(value.get("error_type", value.get("type", "semantic"))),
            action=str(value.get("action", "reopen")),
        )

    def to_dict(self):
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "reason": self.reason,
            "confidence": self.confidence,
            "error_type": self.error_type,
            "action": self.action,
        }


@dataclass
class SemanticEvaluation:
    has_error: bool
    summary: str = ""
    regions: List[RegionSelection] = field(default_factory=list)
    correction_instruction: str = ""
    should_abstain: bool = False
    parser_error: Optional[str] = None
    raw: Optional[Any] = None

    @classmethod
    def abstain(cls, reason, raw=None):
        return cls(False, summary=reason, should_abstain=True, parser_error=reason, raw=raw)

    def actionable_regions(self):
        if self.should_abstain or not self.has_error:
            return []
        return [region for region in self.regions if region.action == "reopen"]

    def to_dict(self):
        return {
            "has_error": self.has_error,
            "summary": self.summary,
            "regions": [region.to_dict() for region in self.regions],
            "correction_instruction": self.correction_instruction,
            "should_abstain": self.should_abstain,
            "parser_error": self.parser_error,
            "raw": self.raw,
        }


@dataclass
class TokenReopenMask:
    token_grid_size: int
    mask: List[List[bool]]

    @classmethod
    def empty(cls, token_grid_size=16):
        return cls(token_grid_size, [[False for _ in range(token_grid_size)] for _ in range(token_grid_size)])

    @classmethod
    def from_indices(cls, indices, token_grid_size=16):
        mask = cls.empty(token_grid_size)
        for row, col in indices:
            if 0 <= row < token_grid_size and 0 <= col < token_grid_size:
                mask.mask[row][col] = True
        return mask

    def selected_indices(self):
        return [(row, col) for row in range(self.token_grid_size) for col in range(self.token_grid_size) if self.mask[row][col]]

    def count(self):
        return len(self.selected_indices())

    def any(self):
        return self.count() > 0

    def to_dict(self):
        return {
            "token_grid_size": self.token_grid_size,
            "selected_count": self.count(),
            "selected_indices": self.selected_indices(),
            "mask": self.mask,
        }


def parse_semantic_evaluation(payload, grid_size=4, max_selected_cells=16):
    raw = payload
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise SchemaError("Semantic evaluation must be an object")
    has_error = bool(payload.get("has_error", payload.get("error_present", False)))
    summary = str(payload.get("summary", payload.get("diagnosis", "")))
    raw_regions = payload.get("regions", payload.get("errors", payload.get("selected_regions", [])))
    if raw_regions is None:
        raw_regions = []
    if isinstance(raw_regions, dict):
        raw_regions = [raw_regions]
    regions = [RegionSelection.from_any(region, grid_size) for region in raw_regions]
    selected = []
    for region in regions:
        selected.extend(region.cells)
    unique = {(cell.row, cell.col) for cell in selected}
    if len(unique) > max_selected_cells:
        raise SchemaError("Semantic evaluation selected too many grid cells")
    if has_error and not regions:
        raise SchemaError("Semantic evaluation reports an error but no regions")
    return SemanticEvaluation(
        has_error=has_error,
        summary=summary,
        regions=regions,
        correction_instruction=str(payload.get("correction_instruction", payload.get("instruction", ""))),
        should_abstain=bool(payload.get("should_abstain", False)),
        raw=raw,
    )


def safe_parse_semantic_evaluation(payload, grid_size=4, max_selected_cells=16):
    try:
        return parse_semantic_evaluation(payload, grid_size=grid_size, max_selected_cells=max_selected_cells)
    except Exception as exc:
        return SemanticEvaluation.abstain(str(exc), raw=payload)

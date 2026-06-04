import json
import re
from pathlib import Path

from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.evaluators.base import SemanticEvaluator
from ascr.generators.mmada_native import MMaDANativeEngine


def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("MMaDA self-evaluation response did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _looks_like_match(text):
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    first = lowered.splitlines()[0].strip()
    negative = ("no", "not match", "does not", "doesn't", "mismatch", "missing", "wrong", "incorrect", "violates", "extra")
    positive = ("yes", "match", "matches", "correct", "satisfies", "no error")
    if first.startswith("yes") or first.startswith(("tag: match", "match")):
        return True
    if first.startswith("no") or first.startswith(("tag: error", "error")):
        return False
    if any(marker in lowered for marker in ("no semantic error", "no error", "fully matches")):
        return True
    if any(marker in lowered for marker in negative):
        return False
    if any(marker in lowered for marker in positive):
        return True
    return None


def _parse_cells_from_text(text, grid_size, max_cells):
    cells = []
    seen = set()
    upper = (text or "").upper()
    for row_text, col_text in re.findall(r"R(\d+)C(\d+)", upper):
        row = int(row_text)
        col = int(col_text)
        if 0 <= row < grid_size and 0 <= col < grid_size and (row, col) not in seen:
            seen.add((row, col))
            cells.append([row, col])
    for row_text, col_text in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text or ""):
        row = int(row_text)
        col = int(col_text)
        if 0 <= row < grid_size and 0 <= col < grid_size and (row, col) not in seen:
            seen.add((row, col))
            cells.append([row, col])
    return cells[:max_cells]


class MMaDASelfEvaluator(SemanticEvaluator):
    """MMaDA-8B evaluating its own generation ("the selector calls itself").

    Uses the *same* loaded :class:`MMaDANativeEngine` as the generator (shared via
    :meth:`attach_engine`) to run image-to-text understanding (``mmu_generate``) in
    two stages, mirroring :class:`ascr.evaluators.showo_mmu.ShowOMMUEvaluator` but at
    direct token-level granularity (``grid_size`` x ``grid_size`` tokens, ``R{row}C{col}``
    coordinates 0-indexed). The selected token cells feed
    :class:`ascr.revision.selector.DirectTokenReopeningSelector` (dilation=0), so MMaDA
    both generates the image and decides exactly which discrete image tokens to reopen.

    No separate evaluator model is loaded: generation and self-evaluation share one 8B
    model. A ``max_selected_cells`` cap is kept as a guardrail (see Phase-7 finding).
    """

    def __init__(self, repo_path="external/MMaDA", checkpoint_path="models/mmada-8b-mixcot", vq_model_path="models/magvitv2", device="cuda", grid_size=32, image_size=512, max_new_tokens=256, max_selected_cells=64, confidence_fallback=True, confidence_fallback_cells=None):
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.vq_model_path = vq_model_path
        self.device = device
        self.grid_size = int(grid_size)
        self.image_size = int(image_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_selected_cells = int(max_selected_cells)
        self.confidence_fallback = bool(confidence_fallback)
        self.confidence_fallback_cells = int(confidence_fallback_cells) if confidence_fallback_cells is not None else int(max_selected_cells)
        self._engine = None

    def attach_engine(self, engine):
        """Share the generator's already-loaded MMaDA engine (load the 8B once)."""
        if isinstance(engine, MMaDANativeEngine):
            self._engine = engine
            return True
        return False

    def _engine_instance(self):
        if self._engine is None:
            self._engine = MMaDANativeEngine(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                vq_model_path=self.vq_model_path,
                device=self.device,
                image_size=self.image_size,
                token_grid_size=self.grid_size,
            )
        return self._engine

    def _clean_image_for(self, grid_image_path):
        """The non-overlaid decoded image that sits beside the grid overlay."""
        path = Path(grid_image_path)
        for name in ("decoded.ppm", "decoded.png"):
            candidate = path.with_name(name)
            if candidate.exists():
                return str(candidate)
        return str(grid_image_path)

    def _confidence_fallback_cells(self, original_prompt, grid_image_path):
        """Let MMaDA judge its own 1024 tokens directly via per-token confidence.

        When the free-form ``R{row}C{col}`` localization fails to ground to the
        token grid, fall back to the model's own confidence: re-encode the clean
        decoded image to its discrete tokens, score every token with
        :meth:`MMaDANativeEngine.token_confidence`, and reopen the lowest-confidence
        cells (capped by ``confidence_fallback_cells``). No down-sampling, no
        external selector: the same 8B model decides which of its own tokens are wrong.
        """
        if not self.confidence_fallback:
            return []
        engine = self._engine_instance()
        score_fn = getattr(engine, "token_confidence", None)
        encode_fn = getattr(engine, "encode_image", None)
        if not callable(score_fn) or not callable(encode_fn):
            return []
        clean_path = self._clean_image_for(grid_image_path)
        try:
            tokens = encode_fn(clean_path)
            confidences = score_fn(original_prompt, tokens)
        except Exception:
            return []
        if not confidences:
            return []
        grid = self.grid_size
        expected = grid * grid
        if len(confidences) != expected:
            return []
        cap = max(1, min(self.confidence_fallback_cells, self.max_selected_cells, expected))
        order = sorted(range(expected), key=lambda i: confidences[i])[:cap]
        return [[idx // grid, idx % grid] for idx in order]

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f"Missing image for MMaDA self-evaluation: {grid_image_path}")
        engine = self._engine_instance()
        eval_question = self._semantic_eval_question(original_prompt)
        eval_text = ""
        try:
            eval_text = engine.answer_image(eval_question, grid_image_path, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            return SemanticEvaluation.abstain(f"MMaDA self-evaluation failed: {exc}", raw={"mmada_eval_text": eval_text})
        match = _looks_like_match(eval_text)
        if match is None:
            match = True
        if match:
            return SemanticEvaluation(
                False,
                summary=str(eval_text).replace("\n", " ")[:300] or "MMaDA judged the image as matching.",
                raw={"mmada_eval_text": eval_text},
            )
        loc_text = ""
        try:
            loc_question = self._localization_question(original_prompt, eval_text)
            loc_text = engine.answer_image(loc_question, grid_image_path, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"MMaDA self-localization failed: {exc}",
                raw={"mmada_eval_text": eval_text, "mmada_localization_text": loc_text},
            )
        raw_cells = []
        loc_payload = None
        try:
            loc_payload = _extract_json_object(loc_text)
            raw_cells = loc_payload.get("grid_cells", loc_payload.get("cells", []))
        except Exception:
            raw_cells = []
        if not raw_cells:
            raw_cells = _parse_cells_from_text(loc_text, self.grid_size, self.max_selected_cells)
        if not raw_cells:
            fallback = self._confidence_fallback_cells(original_prompt, grid_image_path)
            if fallback:
                raw_cells = fallback
                loc_payload = {"source": "self_confidence_fallback", "cells": fallback}
        if not raw_cells:
            return SemanticEvaluation.abstain(
                "MMaDA found a semantic issue but did not localize any token cells.",
                raw={"mmada_eval_text": eval_text, "mmada_localization_text": loc_text},
            )
        try:
            cells = [GridCell.from_any(cell, self.grid_size) for cell in raw_cells]
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"MMaDA returned invalid token cells: {exc}",
                raw={"mmada_eval_text": eval_text, "mmada_localization_text": loc_text},
            )
        unique = []
        seen = set()
        for cell in cells:
            key = (cell.row, cell.col)
            if key not in seen:
                seen.add(key)
                unique.append(cell)
        if len(unique) > self.max_selected_cells:
            unique = unique[:self.max_selected_cells]
        region = RegionSelection(
            cells=unique,
            reason=str(eval_text).replace("\n", " ")[:200],
            confidence=1.0,
            error_type="semantic",
            action="reopen",
        )
        return SemanticEvaluation(
            True,
            summary=str(eval_text).replace("\n", " ")[:300] or "MMaDA found a semantic mismatch.",
            regions=[region],
            correction_instruction="Regenerate the selected tokens to satisfy the original prompt while preserving correct content.",
            raw={"mmada_eval_text": eval_text, "mmada_localization_text": loc_text, "mmada_localization": loc_payload},
        )

    def _coordinate_help(self):
        last = self.grid_size - 1
        return (
            f"The image is overlaid with a {self.grid_size}x{self.grid_size} grid of equally sized cells used "
            f"only as an evaluation aid (do not treat the grid lines as part of the scene). Identify each cell "
            f"by the coordinate R{{row}}C{{col}} where row and col are integers from 0 to {last}; R0C0 is the "
            f"top-left cell and R{last}C{last} is the bottom-right cell."
        )

    def _semantic_eval_question(self, original_prompt):
        return (
            f"Original text-to-image prompt: {original_prompt}. "
            "Carefully compare the image against the prompt (objects, counts, colors, attributes, text, and "
            "spatial relations). Does the image fully satisfy the prompt? Answer with 'yes' or 'no' first, then "
            "give a brief reason."
        )

    def _localization_question(self, original_prompt, issue_text):
        issue = " ".join(str(issue_text).split())[:200]
        last = self.grid_size - 1
        return (
            f"Original text-to-image prompt: {original_prompt}. Identified problem: {issue}. "
            + self._coordinate_help()
            + f" List the R{{row}}C{{col}} cells (at most {self.max_selected_cells}) that tightly cover the wrong, "
            f"missing, or extra content. Pick the smallest set of cells that still covers the error. Answer with the "
            f"cell coordinates such as R{0}C{0}, R{last}C{last}."
        )

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
        raise ValueError("MMaDA coarse self-evaluation response did not contain a JSON object")
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


def _parse_letter_cells(text, grid_size):
    """Parse A1..D4 style coarse-grid labels and [row,col] pairs into cells.

    Column-letter rows follow the Show-o MMU convention: letter = row (A=0),
    number = 1-indexed column. Both ``A1`` and explicit ``[row, col]`` are accepted.
    """
    cells = []
    seen = set()
    upper = (text or "").upper()
    for label, number in re.findall(r"\b([A-Z])(\d+)\b", upper):
        row = ord(label) - ord("A")
        col = int(number) - 1
        if 0 <= row < grid_size and 0 <= col < grid_size and (row, col) not in seen:
            seen.add((row, col))
            cells.append([row, col])
    for row_text, col_text in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text or ""):
        row = int(row_text)
        col = int(col_text)
        if 0 <= row < grid_size and 0 <= col < grid_size and (row, col) not in seen:
            seen.add((row, col))
            cells.append([row, col])
    return cells


class MMaDASelfCoarseEvaluator(SemanticEvaluator):
    """MMaDA-8B judging its own generation at a COARSE grid ("selector calls itself").

    This is the Phase-9 counterpart to :class:`ascr.evaluators.mmada_self.MMaDASelfEvaluator`:
    instead of asking MMaDA to localize wrong cells at the full 32x32 token grid (which it
    cannot ground), it follows the *original* ASCR coarse strategy. MMaDA's MMU understanding
    replaces the Qwen-9B selector and answers at a small ``grid_size`` x ``grid_size`` grid
    (default 4x4, labels ``A1``-``D4``). The selected coarse cells are then projected to the
    32x32 token grid with dilation by :class:`ascr.revision.selector.GridSemanticReopeningSelector`.

    The *same* loaded :class:`MMaDANativeEngine` serves both generation and evaluation (shared
    via :meth:`use_engine`), so the 8B is loaded once. A coarse confidence fallback (per-token
    confidence average-pooled into coarse cells) keeps the loop functional when the free-form
    MMU localization yields no cells.
    """

    def __init__(self, repo_path="external/MMaDA", checkpoint_path="models/mmada-8b-mixcot", vq_model_path="models/magvitv2", device="cuda", grid_size=4, token_grid_size=32, image_size=512, max_new_tokens=48, max_selected_cells=6, confidence_fallback=True, confidence_fallback_cells=None):
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.vq_model_path = vq_model_path
        self.device = device
        self.grid_size = int(grid_size)
        self.token_grid_size = int(token_grid_size)
        self.image_size = int(image_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_selected_cells = int(max_selected_cells)
        self.confidence_fallback = bool(confidence_fallback)
        self.confidence_fallback_cells = int(confidence_fallback_cells) if confidence_fallback_cells is not None else int(max_selected_cells)
        self._engine = None

    def use_engine(self, engine):
        """Share the generator's already-loaded MMaDA engine (load the 8B once)."""
        if isinstance(engine, MMaDANativeEngine):
            self._engine = engine
            return True
        return False

    # Kept for parity with the direct evaluator's wiring helper name.
    attach_engine = use_engine

    def _engine_instance(self):
        if self._engine is None:
            self._engine = MMaDANativeEngine(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                vq_model_path=self.vq_model_path,
                device=self.device,
                image_size=self.image_size,
                token_grid_size=self.token_grid_size,
            )
        return self._engine

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f"Missing image for MMaDA coarse self-evaluation: {grid_image_path}")
        engine = self._engine_instance()
        eval_question = self._semantic_eval_question(original_prompt)
        eval_text = ""
        try:
            eval_text = engine.answer_image(eval_question, grid_image_path, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            return SemanticEvaluation.abstain(f"MMaDA coarse self-evaluation failed: {exc}", raw={"mmada_eval_text": eval_text})
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
                f"MMaDA coarse self-localization failed: {exc}",
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
            raw_cells = _parse_letter_cells(loc_text, self.grid_size)
        if not raw_cells:
            fallback = self._confidence_fallback_cells(original_prompt, grid_image_path)
            if fallback:
                raw_cells = fallback
                loc_payload = {"source": "self_confidence_coarse_fallback", "cells": fallback}
        if not raw_cells:
            return SemanticEvaluation.abstain(
                "MMaDA found a semantic issue but did not localize any coarse cells.",
                raw={"mmada_eval_text": eval_text, "mmada_localization_text": loc_text},
            )
        try:
            cells = [GridCell.from_any(cell, self.grid_size) for cell in raw_cells]
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"MMaDA returned invalid coarse cells: {exc}",
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
            correction_instruction="Regenerate the selected region to satisfy the original prompt while preserving correct content.",
            raw={"mmada_eval_text": eval_text, "mmada_localization_text": loc_text, "mmada_localization": loc_payload},
        )

    def _clean_image_for(self, grid_image_path):
        path = Path(grid_image_path)
        for name in ("decoded.ppm", "decoded.png"):
            candidate = path.with_name(name)
            if candidate.exists():
                return str(candidate)
        return str(grid_image_path)

    def _confidence_fallback_cells(self, original_prompt, grid_image_path):
        """Average-pool MMaDA's own per-token confidence into coarse cells.

        Re-encodes the clean decoded image, scores all 32x32 tokens with
        :meth:`MMaDANativeEngine.token_confidence`, average-pools confidence into the
        ``grid_size`` x ``grid_size`` coarse grid, and returns the lowest-confidence
        coarse cells (capped). Keeps the coarse loop functional when free-form
        localization fails, while staying faithful to "the model judges its own tokens".
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
        tg = self.token_grid_size
        if not confidences or len(confidences) != tg * tg:
            return []
        cg = self.grid_size
        if tg % cg != 0:
            return []
        factor = tg // cg
        sums = [0.0] * (cg * cg)
        counts = [0] * (cg * cg)
        for idx, conf in enumerate(confidences):
            trow = idx // tg
            tcol = idx % tg
            cidx = (trow // factor) * cg + (tcol // factor)
            sums[cidx] += float(conf)
            counts[cidx] += 1
        means = [sums[i] / counts[i] if counts[i] else 1.0 for i in range(cg * cg)]
        cap = max(1, min(self.confidence_fallback_cells, self.max_selected_cells, cg * cg))
        order = sorted(range(cg * cg), key=lambda i: means[i])[:cap]
        return [[idx // cg, idx % cg] for idx in order]

    def _coordinate_help(self):
        last_letter = chr(ord("A") + self.grid_size - 1)
        return (
            f"The image is overlaid with a {self.grid_size}x{self.grid_size} grid whose cells are "
            f"labelled A1 (top-left) through {last_letter}{self.grid_size} (bottom-right); the letter is the "
            f"row (A is the top row) and the number is the column (1 is the left column)."
        )

    def _semantic_eval_question(self, original_prompt):
        return (
            f"Original text-to-image prompt: {original_prompt}. "
            "Does the image fully satisfy the prompt (objects, counts, colors, attributes, and spatial "
            "relations)? Answer 'yes' or 'no' first, then give a brief reason."
        )

    def _localization_question(self, original_prompt, issue_text):
        issue = " ".join(str(issue_text).split())[:200]
        last_letter = chr(ord("A") + self.grid_size - 1)
        return (
            f"Original text-to-image prompt: {original_prompt}. Identified problem: {issue}. "
            + self._coordinate_help()
            + f" Which grid cells contain the wrong, missing, or extra content? Answer with at most "
            f"{self.max_selected_cells} cell labels such as A1, B2, {last_letter}{self.grid_size}, then a short reason."
        )

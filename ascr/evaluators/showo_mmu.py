import json
import re
from pathlib import Path

from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.evaluators.base import SemanticEvaluator
from ascr.generators.showo_native import ShowONativeEngine


def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Show-o MMU response did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _fallback_semantic_payload(text):
    normalized = (text or "").strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    summary = normalized.replace("\n", " ")[:300]
    first_line = lowered.splitlines()[0].strip() if lowered.splitlines() else lowered
    positive_markers = (
        "tag: match",
        "match",
        "matches",
        "match the prompt",
        "satisfies",
        "satisfy the prompt",
        "correct",
        "no semantic error",
        "no error",
        "yes",
    )
    negative_markers = (
        "tag: error",
        "does not match",
        "not match",
        "mismatch",
        "violates",
        "incorrect",
        "wrong",
        "missing",
        "extra",
        "semantic error",
        "error",
        "not satisfy",
    )
    if first_line.startswith(("tag: match", "match", "yes", "correct")) or any(marker in lowered for marker in ("no semantic error", "no error")):
        return {
            "match": True,
            "summary": summary,
            "errors": [],
            "source": "showo_mmu_natural_language_fallback",
        }
    if first_line.startswith(("tag: error", "error", "mismatch", "no")) or any(marker in lowered for marker in negative_markers):
        return {
            "match": False,
            "summary": summary,
            "errors": [{
                "type": "semantic",
                "object": "unknown",
                "issue": summary,
                "severity": "medium",
                "suggested_fix": "Regenerate the selected region to satisfy the original prompt.",
            }],
            "source": "showo_mmu_natural_language_fallback",
        }
    if any(marker in lowered for marker in positive_markers):
        return {
            "match": True,
            "summary": summary,
            "errors": [],
            "source": "showo_mmu_natural_language_fallback",
        }
    return None


def _fallback_localization_payload(text, grid_size):
    normalized = (text or "").strip()
    if not normalized:
        return None
    cells = []
    upper = normalized.upper()
    for label, number in re.findall(r"\b([A-Z])(\d+)\b", upper):
        row = ord(label) - ord("A")
        col = int(number) - 1
        if 0 <= row < grid_size and 0 <= col < grid_size:
            cells.append([row, col])
    for row_text, col_text in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", normalized):
        row = int(row_text)
        col = int(col_text)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            cells.append([row, col])
    unique_cells = []
    for cell in cells:
        if cell not in unique_cells:
            unique_cells.append(cell)
    if not unique_cells:
        return None
    confidence = 1.0
    confidence_match = re.search(r"confidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", normalized, re.IGNORECASE)
    if confidence_match:
        confidence = float(confidence_match.group(1))
    return {
        "grid_cells": unique_cells,
        "localization_rationale": normalized.replace("\n", " ")[:300],
        "confidence": confidence,
        "source": "showo_mmu_natural_language_fallback",
    }


class ShowOMMUEvaluator(SemanticEvaluator):
    def __init__(self, repo_path="external/Show-o", checkpoint_path="models/show-o-512x512", vq_model_path="models/magvitv2", llm_model_path="models/phi-1_5", showo_config_path="configs/showo_local_512x512.yaml", device="cuda", grid_size=4, image_size=512, max_new_tokens=192):
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.vq_model_path = vq_model_path
        self.llm_model_path = llm_model_path
        self.showo_config_path = showo_config_path
        self.device = device
        self.grid_size = int(grid_size)
        self.image_size = int(image_size)
        self.max_new_tokens = int(max_new_tokens)
        self._engine = None

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f"Missing image for Show-o MMU evaluation: {grid_image_path}")
        eval_question = self._semantic_eval_question(original_prompt)
        eval_text = ""
        try:
            eval_text = self._engine_instance().answer_image(eval_question, grid_image_path, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            return SemanticEvaluation.abstain(f"Show-o MMU semantic evaluation failed: {exc}", raw={"showo_eval_text": eval_text})
        try:
            eval_payload = _extract_json_object(eval_text)
            parse_error = None
        except Exception as exc:
            eval_payload = _fallback_semantic_payload(eval_text)
            parse_error = str(exc)
            if eval_payload is None:
                return SemanticEvaluation.abstain(
                    f"Show-o MMU semantic evaluation failed: {exc}",
                    raw={"showo_eval_text": eval_text, "showo_eval_parse_error": str(exc)},
                )
        match = bool(eval_payload.get("match", not bool(eval_payload.get("has_error", False))))
        if match:
            return SemanticEvaluation(
                False,
                summary=str(eval_payload.get("summary", "Show-o MMU judged the image as matching.")),
                raw={"showo_eval_text": eval_text, "showo_eval": eval_payload, "showo_eval_parse_error": parse_error},
            )
        loc_text = ""
        try:
            loc_question = self._localization_question(original_prompt, eval_payload)
            loc_text = self._engine_instance().answer_image(loc_question, grid_image_path, max_new_tokens=self.max_new_tokens)
            try:
                loc_payload = _extract_json_object(loc_text)
                loc_parse_error = None
            except Exception as exc:
                loc_payload = _fallback_localization_payload(loc_text, self.grid_size)
                loc_parse_error = str(exc)
                if loc_payload is None:
                    raise
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"Show-o MMU localization failed: {exc}",
                raw={"showo_eval_text": eval_text, "showo_eval": eval_payload, "showo_eval_parse_error": parse_error, "showo_localization_text": loc_text},
            )
        raw_cells = loc_payload.get("grid_cells", loc_payload.get("cells", []))
        if not raw_cells:
            return SemanticEvaluation.abstain("Show-o MMU found a semantic issue but did not localize grid cells.", raw={"showo_eval_text": eval_text, "showo_eval": eval_payload, "showo_localization_text": loc_text, "showo_localization": loc_payload})
        try:
            cells = [GridCell.from_any(cell, self.grid_size) for cell in raw_cells]
        except Exception as exc:
            return SemanticEvaluation.abstain(f"Show-o MMU returned invalid grid cells: {exc}", raw={"showo_eval_text": eval_text, "showo_eval": eval_payload, "showo_localization_text": loc_text, "showo_localization": loc_payload})
        fix_instruction = self._fix_instruction(eval_payload)
        region = RegionSelection(
            cells=cells,
            reason=str(loc_payload.get("localization_rationale", eval_payload.get("summary", "Show-o localized a semantic mismatch."))),
            confidence=float(loc_payload.get("confidence", 1.0)),
            error_type=str(self._error_type(eval_payload)),
            action="reopen",
        )
        return SemanticEvaluation(
            True,
            summary=str(eval_payload.get("summary", "Show-o MMU found a semantic mismatch.")),
            regions=[region],
            correction_instruction=fix_instruction,
            raw={"showo_eval_text": eval_text, "showo_eval": eval_payload, "showo_eval_parse_error": parse_error, "showo_localization_text": loc_text, "showo_localization": loc_payload, "showo_localization_parse_error": loc_parse_error},
        )

    def _engine_instance(self):
        if self._engine is None:
            self._engine = ShowONativeEngine(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                vq_model_path=self.vq_model_path,
                llm_model_path=self.llm_model_path,
                showo_config_path=self.showo_config_path,
                device=self.device,
                image_size=self.image_size,
                token_grid_size=self.grid_size * 8,
            )
        return self._engine

    def _semantic_eval_question(self, original_prompt):
        return (
            f"Does this image show {original_prompt}? "
            "Answer yes or no."
        )

    def _localization_question(self, original_prompt, eval_payload):
        issue = eval_payload.get("summary", "semantic mismatch")
        return (
            "The image has a visible 4x4 grid with cells A1 through D4. "
            f"Prompt: {original_prompt}. Issue: {issue}. "
            "Which grid cells contain the mismatch? Answer with cell labels such as A1,B2, then a short reason."
        )

    def _fix_instruction(self, eval_payload):
        errors = eval_payload.get("errors") or []
        if errors and isinstance(errors, list) and isinstance(errors[0], dict):
            return str(errors[0].get("suggested_fix", errors[0].get("issue", "Regenerate the selected region to satisfy the prompt.")))
        return str(eval_payload.get("correction_instruction", "Regenerate the selected region to satisfy the original prompt while preserving correct content."))

    def _error_type(self, eval_payload):
        errors = eval_payload.get("errors") or []
        if errors and isinstance(errors, list) and isinstance(errors[0], dict):
            return errors[0].get("type", "semantic")
        return "semantic"

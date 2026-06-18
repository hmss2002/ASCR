import json
from pathlib import Path

from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.training.localizer_model import predict_cells


class StudentLocalizerEvaluator:
    def __init__(self, model_path, threshold=None, max_selected_cells=None):
        if not model_path:
            raise ValueError("student_localizer evaluator requires evaluator.model_path or STUDENT_MODEL")
        self.model_path = Path(model_path)
        self.model = json.loads(self.model_path.read_text(encoding="utf-8"))
        if threshold is not None:
            self.model["threshold"] = float(threshold)
        if max_selected_cells is not None:
            self.model["max_selected_cells"] = int(max_selected_cells)

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        prompt = current_prompt or original_prompt
        selected, scored = predict_cells(self.model, prompt, grid_image_path)
        if not selected:
            return SemanticEvaluation(
                has_error=False,
                summary="Student localizer found no actionable semantic error.",
                regions=[],
                correction_instruction="",
                raw={"model_path": str(self.model_path), "top_scores": scored[:8]},
            )
        cells = [GridCell.from_any(label, int(self.model.get("grid_size", 4))) for label in selected]
        top_score = scored[0][1] if scored else 0.0
        return SemanticEvaluation(
            has_error=True,
            summary=f"Student localizer selected {len(cells)} grid cell(s) for semantic reopening.",
            regions=[
                RegionSelection(
                    cells=cells,
                    reason="student_localizer_v0 predicted semantic mismatch",
                    confidence=max(0.0, min(1.0, float(top_score))),
                    error_type="semantic",
                    action="reopen",
                )
            ],
            correction_instruction="Correct the selected region to better match the original prompt.",
            raw={"model_path": str(self.model_path), "top_scores": scored[:8]},
        )

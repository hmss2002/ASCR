from pathlib import Path

from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.evaluators.base import SemanticEvaluator
from ascr.training.selector_model import LearnedCoarseSelectorModel


class LearnedStage2Evaluator(SemanticEvaluator):
    def __init__(self, checkpoint_path, device="cpu", grid_size=4, max_selected_cells=8, error_threshold=0.5, cell_threshold=0.5):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.grid_size = int(grid_size)
        self.max_selected_cells = int(max_selected_cells)
        self.error_threshold = float(error_threshold)
        self.cell_threshold = float(cell_threshold)
        self._model = None

    def model(self):
        if self._model is None:
            self._model = LearnedCoarseSelectorModel.load(self.checkpoint_path, map_location=self.device)
        return self._model

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f"Missing image for learned Stage 2 evaluation: {grid_image_path}")
        try:
            prediction = self.model().predict(
                original_prompt,
                grid_image_path,
                iteration=iteration,
                device=self.device,
                error_threshold=self.error_threshold,
                cell_threshold=self.cell_threshold,
            )
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"Learned Stage 2 evaluator failed: {exc}",
                raw={"checkpoint_path": self.checkpoint_path},
            )
        cells = [GridCell.from_any(label, self.grid_size) for label in prediction["selected_cells"][:self.max_selected_cells]]
        if not prediction["has_error"] or not cells:
            return SemanticEvaluation(
                has_error=False,
                summary="Learned selector found no material semantic error.",
                regions=[],
                correction_instruction="",
                raw={"prediction": prediction, "checkpoint_path": self.checkpoint_path},
            )
        region = RegionSelection(
            cells=cells,
            reason="learned semantic mismatch",
            confidence=float(prediction["error_probability"]),
            error_type="semantic",
            action="reopen",
        )
        return SemanticEvaluation(
            has_error=True,
            summary="Learned selector predicts a semantic mismatch in the selected cells.",
            regions=[region],
            correction_instruction="Regenerate the selected grid cells so the image better satisfies the original prompt while preserving correct content.",
            raw={"prediction": prediction, "checkpoint_path": self.checkpoint_path},
        )

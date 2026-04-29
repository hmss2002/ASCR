import tempfile
import unittest
from pathlib import Path

from ascr.core.loop import ASCRLoop, ASCRRunConfig
from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.generators.base import MockGeneratorAdapter
from ascr.revision.selector import GridSemanticReopeningSelector


class TwoErrorEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        self.calls.append((iteration, current_prompt, Path(grid_image_path).name))
        if iteration < 2:
            return SemanticEvaluation(
                True,
                summary=f"semantic miss {iteration}",
                regions=[RegionSelection(cells=[GridCell(0, 0)], reason="wrong object", confidence=0.9, error_type="semantic", action="reopen")],
                correction_instruction=f"fix selected region {iteration}",
)
        return SemanticEvaluation(False, summary="ok")


class LoopMultiInsertTests(unittest.TestCase):
    def test_loop_inserts_evaluator_feedback_until_clean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = TwoErrorEvaluator()
            loop = ASCRLoop(
                MockGeneratorAdapter(token_grid_size=4, image_size=64),
                evaluator,
                GridSemanticReopeningSelector(coarse_grid_size=4, token_grid_size=4, dilation=0),
                ASCRRunConfig(run_name="test_multi_insert", max_iterations=4, image_size=64, coarse_grid_size=4, token_grid_size=4, output_dir=temp_dir),
)
            summary = loop.run("complex prompt", project_root=temp_dir)
            artifact_root = Path(summary["artifact_root"])
            self.assertEqual(summary["stop_reason"], "no_semantic_error")
            self.assertEqual(summary["iterations_recorded"], 2)
            self.assertEqual(summary["evaluator_calls"], 3)
            self.assertEqual(len(summary["revision_records"]), 2)
            self.assertEqual([call[0] for call in evaluator.calls], [0, 1, 2])
            self.assertTrue((artifact_root / "iterations/000/correction_prompt.txt").exists())
            self.assertTrue((artifact_root / "iterations/001/correction_prompt.txt").exists())
            self.assertFalse((artifact_root / "iterations/002/correction_prompt.txt").exists())
            trace_lines = (artifact_root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(trace_lines), 3)


if __name__ == "__main__":
    unittest.main()

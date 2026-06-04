import tempfile
import unittest
from pathlib import Path

from ascr.core.loop import ASCRRunConfig
from ascr.core.loop_direct import DirectTokenReopenLoop
from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.generators.base import MockGeneratorAdapter
from ascr.revision.selector import DirectTokenReopeningSelector


class OneErrorTokenEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        self.calls.append(iteration)
        if iteration == 0:
            regions = [RegionSelection(cells=[GridCell(2, 3)], reason="wrong", confidence=0.9, error_type="semantic", action="reopen")]
            return SemanticEvaluation(True, summary="token miss", regions=regions, correction_instruction="fix token")
        return SemanticEvaluation(False, summary="ok")


class DirectLoopTests(unittest.TestCase):
    def test_direct_loop_reopens_exact_tokens_and_stamps_variant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = OneErrorTokenEvaluator()
            loop = DirectTokenReopenLoop(
                MockGeneratorAdapter(token_grid_size=8, image_size=64),
                evaluator,
                DirectTokenReopeningSelector(token_grid_size=8, dilation=0),
                ASCRRunConfig(run_name="test_direct", max_iterations=4, image_size=64, token_grid_size=8, output_dir=temp_dir),
                label_step=2,
            )
            summary = loop.run("a prompt", project_root=temp_dir)
            artifact_root = Path(summary["artifact_root"])
            self.assertEqual(summary["stage1_variant"], "direct_token")
            self.assertEqual(summary["stop_reason"], "no_semantic_error")
            self.assertEqual(summary["iterations_recorded"], 1)
            self.assertEqual(evaluator.calls, [0, 1])
            config_snapshot = (artifact_root / "config_snapshot.json").read_text(encoding="utf-8")
            self.assertIn("direct_token", config_snapshot)
            mask_json = (artifact_root / "iterations/000/reopen_mask.json").read_text(encoding="utf-8")
            self.assertIn("\"selected_count\": 1", mask_json)


if __name__ == "__main__":
    unittest.main()

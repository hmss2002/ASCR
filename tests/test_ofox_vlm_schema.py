import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ascr.evaluators.ofox_vlm import OfoxVLMEvaluator


class OfoxVLMEvaluatorTests(unittest.TestCase):
    def _image_path(self, tmpdir):
        image_path = Path(tmpdir) / "grid.png"
        Image.new("RGB", (64, 64), (10, 10, 10)).save(image_path)
        return image_path

    def test_parses_teacher_json_into_semantic_evaluation(self):
        evaluator = OfoxVLMEvaluator(api_key="test-key", base_url="https://api.ofox.ai/v1")
        payload = json.dumps({
            "has_error": True,
            "summary": "left-right relation is wrong",
            "regions": [{"cells": ["A1", {"row": 1, "col": 2}], "reason": "wrong relation", "confidence": 0.91, "error_type": "spatial", "action": "reopen"}],
            "correction_instruction": "move the red cube left of the blue sphere",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._image_path(tmpdir)
            with patch.object(evaluator, "_ensure_capability_check", return_value=None), \
                    patch.object(evaluator, "_create_completion", return_value=(payload, {"usage": {"total_tokens": 123}})):
                evaluation = evaluator.evaluate("A red cube left of a blue sphere", str(image_path), 0)
        self.assertTrue(evaluation.has_error)
        self.assertFalse(evaluation.should_abstain)
        self.assertEqual([cell.to_label() for cell in evaluation.regions[0].cells], ["A1", "B3"])
        self.assertEqual(evaluation.raw["ofox_payload"]["regions"][0]["cells"][0], "A1")

    def test_repairs_malformed_json_with_same_image_path(self):
        evaluator = OfoxVLMEvaluator(api_key="test-key", base_url="https://api.ofox.ai/v1")
        repaired = json.dumps({
            "has_error": False,
            "summary": "the image matches the prompt",
            "regions": [],
            "correction_instruction": "",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._image_path(tmpdir)
            with patch.object(evaluator, "_ensure_capability_check", return_value=None), \
                    patch.object(evaluator, "_create_completion", side_effect=[
                        ("The image looks good but I forgot the JSON wrapper.", {"stage": "diagnosis"}),
                        (repaired, {"stage": "repair"}),
                    ]) as create_completion:
                evaluation = evaluator.evaluate("A red cube left of a blue sphere", str(image_path), 0)
        self.assertFalse(evaluation.has_error)
        self.assertFalse(evaluation.should_abstain)
        self.assertEqual(evaluation.raw["ofox_repair_text"], repaired)
        self.assertEqual(create_completion.call_count, 2)

    def test_abstains_when_repair_still_fails(self):
        evaluator = OfoxVLMEvaluator(api_key="test-key", base_url="https://api.ofox.ai/v1")
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._image_path(tmpdir)
            with patch.object(evaluator, "_ensure_capability_check", return_value=None), \
                    patch.object(evaluator, "_create_completion", side_effect=[
                        ("not json", {"stage": "diagnosis"}),
                        ("still not json", {"stage": "repair"}),
                    ]):
                evaluation = evaluator.evaluate("A red cube left of a blue sphere", str(image_path), 0)
        self.assertTrue(evaluation.should_abstain)
        self.assertIn("failed", evaluation.summary.lower())


if __name__ == "__main__":
    unittest.main()

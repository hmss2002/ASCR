import json
import unittest

from ascr.evaluators.qwen_vl import QwenVLEvaluator, _extract_json_object, _normalize_payload
from ascr.evaluators.registry import build_evaluator


class QwenVLEvaluatorHelpersTest(unittest.TestCase):
    def test_extract_json_from_fenced_text(self):
        text = "```json" + json.dumps({"has_error": False, "regions": []}) + "``` trailing"
        payload = _extract_json_object(text)
        self.assertFalse(payload["has_error"])

    def test_normalize_match_payload(self):
        payload = _normalize_payload({"match": True, "summary": "ok"})
        self.assertFalse(payload["has_error"])
        self.assertEqual(payload["summary"], "ok")

    def test_normalize_error_grid_cells(self):
        payload = _normalize_payload({"match": False, "summary": "wrong color", "errors": [{"grid_cells": ["B2"], "issue": "wrong color", "type": "attribute"}], "suggested_fix": "make the object red"})
        self.assertTrue(payload["has_error"])
        self.assertEqual(payload["regions"][0]["cells"], ["B2"])
        self.assertEqual(payload["regions"][0]["error_type"], "attribute")
        self.assertEqual(payload["correction_instruction"], "make the object red")

    def test_registry_accepts_qwen_backend(self):
        evaluator = build_evaluator("local_vlm", {"coarse_grid_size": 4, "image_size": 512, "evaluator": {"backend": "qwen3_6", "model_path": "Qwen/Qwen3.6-35B-A3B"}})
        self.assertIsInstance(evaluator, QwenVLEvaluator)
        self.assertEqual(evaluator.model_path, "Qwen/Qwen3.6-35B-A3B")
        self.assertFalse(evaluator.processor_use_fast)


if __name__ == "__main__":
    unittest.main()

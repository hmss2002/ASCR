import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ascr.evaluators.qwen_vl import (
    QwenVLEvaluator,
    _extract_json_object,
    _normalize_payload,
    _require_native_qwen35_moe_support,
)
from ascr.evaluators.registry import build_evaluator


class QwenVLEvaluatorHelpersTest(unittest.TestCase):
    def _patch_torchvision_available(self, value=True):
        try:
            import transformers.utils
        except Exception as exc:
            self.skipTest(f"transformers unavailable: {exc}")
        return patch("transformers.utils.is_torchvision_available", return_value=value)

    def test_extract_json_from_fenced_text(self):
        text = "```json" + json.dumps({"has_error": False, "regions": []}) + "``` trailing"
        payload = _extract_json_object(text)
        self.assertFalse(payload["has_error"])

    def test_extract_json_skips_schema_example(self):
        text = "Use this schema: {\"has_error\": boolean, \"regions\": array}. " + json.dumps({"has_error": False, "summary": "ok", "regions": [], "correction_instruction": ""})
        self.assertFalse(_extract_json_object(text)["has_error"])

    def test_extract_json_prefers_final_answer_after_thinking(self):
        thinking_payload = json.dumps({"has_error": True, "summary": "bad intermediate", "regions": [{"cells": ["A1"]}], "correction_instruction": "fix it"})
        final_payload = json.dumps({"has_error": False, "summary": "ok", "regions": [], "correction_instruction": ""})
        text = f"<think>{thinking_payload}</think>{final_payload}"
        self.assertFalse(_extract_json_object(text)["has_error"])
        marked_text = "Brief check. FINAL_JSON: " + final_payload
        self.assertFalse(_extract_json_object(marked_text)["has_error"])

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

    def test_normalize_payload_caps_cells_round_robin(self):
        payload = _normalize_payload({"has_error": True, "regions": [{"cells": ["A1", "A2", "A3"], "reason": "first"}, {"cells": ["B1", "B2"], "reason": "second"}, {"cells": ["C1"], "reason": "third"}]}, max_selected_cells=4)
        self.assertEqual([region["cells"] for region in payload["regions"]], [["A1", "A2"], ["B1"], ["C1"]])

    def test_question_demands_json_only(self):
        question = QwenVLEvaluator()._build_question("A red cube left of a blue sphere")
        self.assertNotIn("/no_think", question)
        self.assertIn("FINAL_JSON:", question)
        self.assertIn("under 80 words", question)
        self.assertIn("close it with </think>", question)
        self.assertIn("After </think>", question)
        self.assertIn("Stop immediately after", question)

    def test_question_can_disable_thinking(self):
        question = QwenVLEvaluator(enable_thinking=False)._build_question("A red cube left of a blue sphere")
        self.assertIn("/no_think", question)
        self.assertIn("Return exactly one valid JSON object", question)
        self.assertIn("Do not include prose or analysis", question)

    def test_evaluate_repairs_non_json_response(self):
        evaluator = QwenVLEvaluator(model_path="local-qwen", strict_json=True, max_new_tokens=64, repair_max_new_tokens=512)
        raw_text = "The red cube is to the left of the blue sphere, so the image satisfies the prompt."
        repaired = json.dumps({"has_error": False, "summary": "The relation is correct.", "regions": [], "correction_instruction": ""})
        with patch.object(evaluator, "_generate_text", side_effect=[raw_text, repaired]) as generate:
            evaluation = evaluator.evaluate("A red cube left of a blue sphere", Path(__file__), 0)
        self.assertFalse(evaluation.should_abstain)
        self.assertFalse(evaluation.has_error)
        self.assertEqual(evaluation.raw["qwen_vl_json_text"], repaired)
        self.assertEqual(generate.call_args_list[1].kwargs["max_new_tokens"], 512)

    def test_repair_budget_defaults_to_at_least_384(self):
        evaluator = QwenVLEvaluator(max_new_tokens=128)
        self.assertEqual(evaluator.repair_max_new_tokens, 384)
        evaluator = QwenVLEvaluator(max_new_tokens=768)
        self.assertEqual(evaluator.repair_max_new_tokens, 768)

    def test_apply_chat_template_enables_thinking_by_default(self):
        class Processor:
            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "template"

        evaluator = QwenVLEvaluator()
        evaluator._processor = Processor()
        self.assertEqual(evaluator._apply_chat_template([], tokenize=False), "template")
        self.assertIs(evaluator._processor.kwargs["enable_thinking"], True)

    def test_apply_chat_template_disables_thinking_when_supported(self):
        class Processor:
            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "template"

        evaluator = QwenVLEvaluator(enable_thinking=False)
        evaluator._processor = Processor()
        self.assertEqual(evaluator._apply_chat_template([], tokenize=False), "template")
        self.assertIs(evaluator._processor.kwargs["enable_thinking"], False)

    def test_apply_chat_template_falls_back_for_legacy_processors(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                if "enable_thinking" in kwargs:
                    raise TypeError("unexpected keyword argument enable_thinking")
                return kwargs

        evaluator = QwenVLEvaluator(enable_thinking=False)
        evaluator._processor = Processor()
        self.assertNotIn("enable_thinking", evaluator._apply_chat_template([], tokenize=False))

    def test_qwen35_moe_rejects_compat_config(self):
        CompatConfig = type("Qwen35MoeCompatConfig", (), {"model_type": "qwen3_5_moe"})
        AutoModel = type("AutoModel", (), {"_model_mapping": {}})
        with self.assertRaises(RuntimeError) as ctx:
            _require_native_qwen35_moe_support(CompatConfig(), AutoModel)
        self.assertIn("native Transformers support", str(ctx.exception))
        self.assertIn("Qwen3_5MoeConfig", str(ctx.exception))

    def test_qwen35_moe_rejects_wrong_model_mapping(self):
        NativeConfig = type("Qwen3_5MoeConfig", (), {"model_type": "qwen3_5_moe"})
        WrongModel = type("Qwen3VLMoeForConditionalGeneration", (), {})
        AutoModel = type("AutoModel", (), {"_model_mapping": {NativeConfig: WrongModel}})
        with self.assertRaises(RuntimeError) as ctx:
            _require_native_qwen35_moe_support(NativeConfig(), AutoModel)
        self.assertIn("Qwen3_5MoeForConditionalGeneration", str(ctx.exception))

    def test_qwen35_moe_requires_torchvision_for_processor(self):
        NativeConfig = type("Qwen3_5MoeConfig", (), {"model_type": "qwen3_5_moe"})
        NativeModel = type("Qwen3_5MoeForConditionalGeneration", (), {})
        AutoModel = type("AutoModel", (), {"_model_mapping": {NativeConfig: NativeModel}})
        with self._patch_torchvision_available(False), self.assertRaises(RuntimeError) as ctx:
            _require_native_qwen35_moe_support(NativeConfig(), AutoModel)
        self.assertIn("torchvision is required", str(ctx.exception))

    def test_qwen35_moe_accepts_native_mapping(self):
        NativeConfig = type("Qwen3_5MoeConfig", (), {"model_type": "qwen3_5_moe"})
        NativeModel = type("Qwen3_5MoeForConditionalGeneration", (), {})
        AutoModel = type("AutoModel", (), {"_model_mapping": {NativeConfig: NativeModel}})
        with self._patch_torchvision_available(True):
            _require_native_qwen35_moe_support(NativeConfig(), AutoModel)

    def test_registry_accepts_qwen_backend(self):
        evaluator = build_evaluator("local_vlm", {"coarse_grid_size": 4, "image_size": 512, "evaluator": {"backend": "qwen3_6", "model_path": "Qwen/Qwen3.6-35B-A3B"}})
        self.assertIsInstance(evaluator, QwenVLEvaluator)
        self.assertEqual(evaluator.model_path, "Qwen/Qwen3.6-35B-A3B")
        self.assertFalse(evaluator.processor_use_fast)
        self.assertTrue(evaluator.enable_thinking)

    def test_registry_can_disable_qwen_thinking(self):
        evaluator = build_evaluator("local_vlm", {"coarse_grid_size": 4, "image_size": 512, "evaluator": {"backend": "qwen3_6", "model_path": "Qwen/Qwen3.6-35B-A3B", "enable_thinking": False, "repair_max_new_tokens": 512}})
        self.assertIsInstance(evaluator, QwenVLEvaluator)
        self.assertFalse(evaluator.enable_thinking)
        self.assertEqual(evaluator.repair_max_new_tokens, 512)


if __name__ == "__main__":
    unittest.main()

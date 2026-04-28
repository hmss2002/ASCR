import unittest

from ascr.evaluators.registry import build_evaluator
from ascr.evaluators.showo_mmu import ShowOMMUEvaluator, _extract_json_object, _fallback_localization_payload, _fallback_semantic_payload
from ascr.generators.showo import ShowOAdapter
from ascr.generators.showo_native import compact_token_payload, flat_to_grid


class NativeShowOHelpersTest(unittest.TestCase):
    def test_flat_to_grid(self):
        self.assertEqual(flat_to_grid(list(range(9)), 3), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    def test_compact_token_payload(self):
        payload = compact_token_payload({
            "model_tokens": [1],
            "decoded_tokens": [2],
            "confidence": [0.5],
            "confidence_steps": 2,
            "step_records": [{"step": 0}],
            "mask_token_id": 999,
        })
        self.assertEqual(sorted(payload.keys()), ["confidence_steps", "decoded_tokens", "mask_token_id", "model_tokens", "step_records"])

    def test_state_from_payload_keeps_native_metadata(self):
        adapter = ShowOAdapter(token_grid_size=2, native_token_loop=True)
        state = adapter._state_from_payload(
            "prompt",
            1,
            {
                "model_tokens": [9, 8, 7, 6],
                "decoded_tokens": [1, 2, 3, 4],
                "confidence": [0.1, 0.2, 0.3, 0.4],
                "confidence_mask": [False, True, False, True],
                "confidence_steps": 2,
                "step_records": [{"step": 0}],
                "mask_token_id": 999,
            },
            "image.png",
        )
        self.assertEqual(state.token_grid, [[1, 2], [3, 4]])
        self.assertTrue(state.metadata["native_token_loop"])
        self.assertEqual(state.metadata["confidence_remask_count"], 2)


class ShowOMMUEvaluatorTest(unittest.TestCase):
    def test_extract_json_object_from_text(self):
        self.assertEqual(_extract_json_object("prefix {\"match\": false} suffix"), {"match": False})

    def test_fallback_semantic_payload_positive_text(self):
        payload = _fallback_semantic_payload("The image matches the prompt correctly.")
        self.assertTrue(payload["match"])

    def test_fallback_semantic_payload_negative_text(self):
        payload = _fallback_semantic_payload("The image does not match because an object is missing.")
        self.assertFalse(payload["match"])
        self.assertEqual(payload["errors"][0]["type"], "semantic")

    def test_fallback_semantic_payload_no_answer_is_negative(self):
        payload = _fallback_semantic_payload("No, the blue sphere is missing")
        self.assertFalse(payload["match"])

    def test_fallback_semantic_payload_no_error_is_positive(self):
        payload = _fallback_semantic_payload("TAG: MATCH\nREASON: no semantic error is visible")
        self.assertTrue(payload["match"])

    def test_fallback_localization_payload_labels(self):
        payload = _fallback_localization_payload("CELLS: A1, C4\nCONFIDENCE: 0.7", 4)
        self.assertEqual(payload["grid_cells"], [[0, 0], [2, 3]])
        self.assertEqual(payload["confidence"], 0.7)

    def test_registry_accepts_showo_mmu_backend(self):
        evaluator = build_evaluator(
            "local_vlm",
            {
                "coarse_grid_size": 4,
                "image_size": 512,
                "generator": {"repo_path": "external/Show-o"},
                "evaluator": {"backend": "showo_mmu"},
            },
        )
        self.assertIsInstance(evaluator, ShowOMMUEvaluator)


if __name__ == "__main__":
    unittest.main()

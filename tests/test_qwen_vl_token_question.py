import unittest

from ascr.evaluators.qwen_vl_token import QwenVLTokenEvaluator


class QwenVLTokenQuestionTests(unittest.TestCase):
    def _evaluator(self, **kwargs):
        return QwenVLTokenEvaluator(select_grid_size=32, max_selected_cells=64, enable_thinking=False, **kwargs)

    def test_grid_size_is_token_resolution(self):
        evaluator = self._evaluator()
        self.assertEqual(evaluator.grid_size, 32)
        self.assertEqual(evaluator.select_grid_size, 32)

    def test_question_uses_numeric_coordinate_scheme(self):
        question = self._evaluator()._build_question("a red cube left of a blue sphere")
        self.assertIn("R{row}C{col}", question)
        self.assertIn("32x32", question)
        self.assertIn("0 to 31", question)
        self.assertIn("64", question)
        self.assertNotIn("A1", question)

    def test_repair_question_uses_numeric_scheme(self):
        repair = self._evaluator()._build_json_repair_question("a prompt", "partial text")
        self.assertIn("R{row}C{col}", repair)
        self.assertIn("0 to 31", repair)
        self.assertIn("has_error", repair)

    def test_thinking_mode_inserts_final_json_marker(self):
        evaluator = QwenVLTokenEvaluator(select_grid_size=32, max_selected_cells=32, enable_thinking=True)
        question = evaluator._build_question("p")
        self.assertIn("FINAL_JSON:", question)
        self.assertIn("<think>", question)

    def test_intermediate_grid_size_label(self):
        evaluator = QwenVLTokenEvaluator(select_grid_size=16, enable_thinking=False)
        question = evaluator._build_question("p")
        self.assertIn("16x16", question)
        self.assertIn("0 to 15", question)


if __name__ == "__main__":
    unittest.main()

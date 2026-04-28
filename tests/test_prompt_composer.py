import unittest

from ascr.core.schemas import SemanticEvaluation
from ascr.revision.prompt_composer import compose_correction_prompt


class PromptComposerTests(unittest.TestCase):
    def test_uses_correction_instruction(self):
        evaluation = SemanticEvaluation(True, correction_instruction="fix cell B2")
        prompt = compose_correction_prompt("original prompt", evaluation)
        self.assertIn("original prompt", prompt)
        self.assertIn("fix cell B2", prompt)

    def test_falls_back_to_summary(self):
        evaluation = SemanticEvaluation(True, summary="wrong count")
        prompt = compose_correction_prompt("prompt", evaluation)
        self.assertIn("wrong count", prompt)


if __name__ == "__main__":
    unittest.main()

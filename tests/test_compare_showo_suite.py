import tempfile
import unittest
from pathlib import Path

from ascr.cli.compare_showo_ascr import build_suite, load_prompts, resolve_ascr_start_mode, suite_to_markdown


class CompareShowoSuiteTests(unittest.TestCase):
    def test_load_prompts_file_ignores_comments_and_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompts.txt"
            path.write_text("# comment\nfirst prompt\n\n second prompt \n", encoding="utf-8")
            self.assertEqual(load_prompts("fallback", path, limit=1), ["first prompt"])
            self.assertEqual(load_prompts("fallback", path), ["first prompt", "second prompt"])

    def test_resolve_ascr_start_mode_prefers_override(self):
        self.assertEqual(resolve_ascr_start_mode({"ascr_start_mode": "partial"}), "partial")
        self.assertEqual(resolve_ascr_start_mode({"ascr_start_mode": "partial"}, "baseline"), "baseline")
        with self.assertRaises(ValueError):
            resolve_ascr_start_mode({"ascr_start_mode": "late"})

    def test_suite_summary_counts_insertions(self):
        result = {
            "prompt": "complex prompt",
            "ascr_start_mode": "partial",
            "evaluator_calls": 3,
            "ascr_insertions": 2,
            "ascr_summary": {"stop_reason": "no_semantic_error"},
            "comparison": {"baseline_score": 0.2, "ascr_score": 0.5, "delta": 0.3, "verdict": "ascr_improved"},
        }
        suite = build_suite([result])
        self.assertEqual(suite["total_evaluator_calls"], 3)
        self.assertEqual(suite["total_ascr_insertions"], 2)
        markdown = suite_to_markdown(suite)
        self.assertIn("complex prompt", markdown)
        self.assertIn("partial", markdown)


if __name__ == "__main__":
    unittest.main()

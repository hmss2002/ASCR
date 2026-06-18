import json
from pathlib import Path
import tempfile
import unittest

from ascr.evaluators.lumina_native import LuminaNativeEvaluator
from ascr.generators.base import _write_mock_ppm
from ascr.training.train_lumina_evaluator import prepare_sft_dataset


class _UnsupportedEngine:
    pass


class _JsonAnswerEngine:
    def answer_image(self, question, image_path, max_new_tokens=384):
        return json.dumps({
            "has_error": True,
            "summary": "object relation mismatch",
            "regions": [{
                "cells": [{"label": "B2"}],
                "reason": "wrong relation",
                "confidence": 0.8,
                "error_type": "relation",
                "action": "reopen",
            }],
            "correction_instruction": "Fix the selected relation.",
        })


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


class LuminaNativeStage2Tests(unittest.TestCase):
    def test_unsupported_native_evaluator_abstains(self):
        evaluator = LuminaNativeEvaluator(engine=_UnsupportedEngine())
        evaluation = evaluator.evaluate("a red cube", "missing.png", 0)
        self.assertFalse(evaluation.has_error)
        self.assertTrue(evaluation.should_abstain)
        self.assertIn("does not expose", evaluation.summary)

    def test_native_evaluator_parses_json_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "grid.ppm"
            _write_mock_ppm(image_path, [[row + col for col in range(4)] for row in range(4)], image_size=32)
            evaluator = LuminaNativeEvaluator(engine=_JsonAnswerEngine(), grid_size=4, max_selected_cells=2)
            evaluation = evaluator.evaluate("a red cube left of a blue sphere", str(image_path), 0)
            self.assertTrue(evaluation.has_error)
            self.assertEqual(evaluation.regions[0].cells[0].to_label(), "B2")
            self.assertEqual(evaluation.raw["method"], "answer_image")

    def test_prepare_lumina_sft_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "images"
            image_root.mkdir()
            _write_mock_ppm(image_root / "grid.ppm", [[row + col for col in range(4)] for row in range(4)], image_size=32)
            dataset = root / "dataset.jsonl"
            write_jsonl(dataset, [{
                "sample_id": "p000",
                "prompt": "a red cube left of a blue sphere",
                "localizations": [{
                    "sample_id": "p000:i000",
                    "prompt": "a red cube left of a blue sphere",
                    "grid_image": "grid.ppm",
                    "evaluation": {
                        "has_error": True,
                        "summary": "wrong relation",
                        "regions": [{"cells": [{"label": "B2"}], "reason": "wrong relation"}],
                        "correction_instruction": "Fix the relation.",
                    },
                }],
            }])
            output = root / "out"
            manifest = prepare_sft_dataset(dataset, output, image_root=image_root, limit=5)
            self.assertEqual(manifest["example_count"], 1)
            rows = [json.loads(line) for line in (output / "sft_examples.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("Return exactly one compact JSON object", rows[0]["input_text"])
            self.assertEqual(rows[0]["target_json"]["regions"][0]["cells"][0]["label"], "B2")


if __name__ == "__main__":
    unittest.main()

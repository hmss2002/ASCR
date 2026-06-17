import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ascr.evaluators.registry import build_evaluator
from ascr.training.train_selector import main


class Stage2TrainingReplayTests(unittest.TestCase):
    def test_dataset_replay_training_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.jsonl"
            dataset_path.write_text(json.dumps({
                "schema_version": "stage2.teacher_trace.v1",
                "prompt": "A red cube left of a blue sphere",
                "iteration": 0,
                "decoded_image": "iterations/000/decoded.ppm",
                "grid_image": "iterations/000/grid.ppm",
                "teacher_json": {"has_error": True},
                "selected_4x4_cells": ["A2"],
                "projected_token_mask": {"selected_count": 16, "token_grid_size": 64},
                "correction_instruction": "move the red cube left",
                "after_image": "iterations/001/decoded.ppm",
            }) + "\n", encoding="utf-8")
            output_dir = Path(tmpdir) / "checkpoints"
            rc = main(["--dataset", str(dataset_path), "--output-dir", str(output_dir)])

            checkpoint = json.loads((output_dir / "selector_checkpoint.json").read_text(encoding="utf-8"))
            replay_lines = (output_dir / "replay_index.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(rc, 0)
        self.assertEqual(checkpoint["model_type"], "dataset_replay")
        self.assertEqual(len(replay_lines), 1)

    def test_learned_coarse_training_writes_checkpoint_and_can_infer(self):
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_a = tmp / "grid_a.png"
            image_b = tmp / "grid_b.png"
            Image.new("RGB", (64, 64), (255, 0, 0)).save(image_a)
            Image.new("RGB", (64, 64), (0, 0, 255)).save(image_b)
            dataset_path = tmp / "dataset.jsonl"
            rows = [
                {
                    "schema_version": "stage2.teacher_trace.v1",
                    "prompt": "a red cube on the left",
                    "iteration": 0,
                    "decoded_image": "iterations/000/decoded.ppm",
                    "grid_image": str(image_a),
                    "teacher_json": {"has_error": True},
                    "selected_4x4_cells": ["A1"],
                    "projected_token_mask": {"selected_count": 16, "token_grid_size": 64},
                    "correction_instruction": "fix the left object",
                },
                {
                    "schema_version": "stage2.teacher_trace.v1",
                    "prompt": "a blue sphere on the right",
                    "iteration": 0,
                    "decoded_image": "iterations/001/decoded.ppm",
                    "grid_image": str(image_b),
                    "teacher_json": {"has_error": True},
                    "selected_4x4_cells": ["D4"],
                    "projected_token_mask": {"selected_count": 16, "token_grid_size": 64},
                    "correction_instruction": "fix the right object",
                },
            ]
            dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output_dir = tmp / "learned"
            rc = main([
                "--dataset", str(dataset_path),
                "--output-dir", str(output_dir),
                "--mode", "learned_coarse",
                "--epochs", "1",
                "--batch-size", "1",
                "--device", "cpu",
            ])
            checkpoint = json.loads((output_dir / "selector_checkpoint.json").read_text(encoding="utf-8"))
            evaluator = build_evaluator("stage2_student", {
                "coarse_grid_size": 4,
                "evaluator": {"checkpoint_path": str(output_dir / "selector_checkpoint.json"), "device": "cpu"},
            })
            evaluation = evaluator.evaluate("a red cube on the left", str(image_a), 0)

        self.assertEqual(rc, 0)
        self.assertEqual(checkpoint["model_type"], "learned_coarse_selector")
        self.assertFalse(evaluation.should_abstain)
        self.assertIn("prediction", evaluation.raw)


if __name__ == "__main__":
    unittest.main()

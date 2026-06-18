import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from ascr.benchmarks.api_image_judge import normalize_judgment, prune_resolved_errors, rewrite_latest_rows
from ascr.benchmarks.image_quality_benchmark import run_benchmark
from ascr.core.loop import ASCRLoop, ASCRRunConfig
from ascr.distill.teacher import extract_json_object
from ascr.evaluators.student_localizer import StudentLocalizerEvaluator
from ascr.generators.base import MockGeneratorAdapter, _write_mock_ppm
from ascr.revision.selector import GridSemanticReopeningSelector
from ascr.training.train_localizer import train_grid_localizer_v0


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


class Args:
    pass


class StudentLocalizerPipelineTests(unittest.TestCase):
    def make_dataset(self, root):
        image_root = Path(root) / "images"
        image_root.mkdir()
        for index in range(4):
            _write_mock_ppm(image_root / f"grid{index}.ppm", [[index + row + col for col in range(4)] for row in range(4)], image_size=32)
        dataset = Path(root) / "dataset.jsonl"
        rows = [
            {
                "sample_id": "p000",
                "prompt": "red cube left of blue sphere",
                "localizations": [{
                    "sample_id": "p000:i000",
                    "prompt": "red cube left of blue sphere",
                    "grid_image": "grid0.ppm",
                    "evaluation": {"has_error": True, "regions": [{"cells": [{"label": "A1"}]}]},
                }],
            },
            {
                "sample_id": "p001",
                "prompt": "green bench and blue bowl",
                "localizations": [{
                    "sample_id": "p001:i000",
                    "prompt": "green bench and blue bowl",
                    "grid_image": "grid1.ppm",
                    "evaluation": {"has_error": False, "regions": []},
                }],
            },
            {
                "sample_id": "p002",
                "prompt": "red cube near blue sphere",
                "localizations": [{
                    "sample_id": "p002:i000",
                    "prompt": "red cube near blue sphere",
                    "grid_image": "grid2.ppm",
                    "evaluation": {"has_error": True, "regions": [{"cells": [{"label": "B2"}]}]},
                }],
            },
            {
                "sample_id": "p003",
                "prompt": "plain landscape",
                "localizations": [{
                    "sample_id": "p003:i000",
                    "prompt": "plain landscape",
                    "grid_image": "grid3.ppm",
                    "evaluation": {"has_error": False, "regions": []},
                }],
            },
        ]
        write_jsonl(dataset, rows)
        return dataset, image_root

    def test_train_localizer_writes_model_and_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, image_root = self.make_dataset(temp_dir)
            result = train_grid_localizer_v0(dataset, image_root, Path(temp_dir) / "student", eval_mode="holdout", seed=7)
            output_dir = Path(result["output_dir"])
            self.assertTrue((output_dir / "student_model.json").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "predictions.jsonl").exists())
            self.assertTrue((output_dir / "split_manifest.json").exists())
            self.assertTrue((output_dir / "holdout_prompts.txt").exists())

    def test_student_evaluator_outputs_semantic_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, image_root = self.make_dataset(temp_dir)
            train_grid_localizer_v0(dataset, image_root, Path(temp_dir) / "student", eval_mode="resubstitution")
            evaluator = StudentLocalizerEvaluator(Path(temp_dir) / "student" / "student_model.json", threshold=-999.0, max_selected_cells=2)
            evaluation = evaluator.evaluate("red cube left of blue sphere", str(image_root / "grid0.ppm"), 0)
            self.assertTrue(evaluation.has_error)
            self.assertLessEqual(len(evaluation.regions[0].cells), 2)

    def test_ascr_loop_uses_student_for_multiple_iterations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, image_root = self.make_dataset(temp_dir)
            train_grid_localizer_v0(dataset, image_root, Path(temp_dir) / "student", eval_mode="resubstitution")
            evaluator = StudentLocalizerEvaluator(Path(temp_dir) / "student" / "student_model.json", threshold=-999.0, max_selected_cells=1)
            loop = ASCRLoop(
                MockGeneratorAdapter(token_grid_size=4, image_size=32),
                evaluator,
                GridSemanticReopeningSelector(coarse_grid_size=4, token_grid_size=4, dilation=0),
                ASCRRunConfig(run_name="student_loop", max_iterations=2, image_size=32, coarse_grid_size=4, token_grid_size=4, output_dir=str(Path(temp_dir) / "runs")),
            )
            summary = loop.run("red cube left of blue sphere", project_root=temp_dir)
            self.assertEqual(summary["evaluator_calls"], 2)
            self.assertEqual(summary["iterations_recorded"], 2)
            self.assertEqual(summary["stop_reason"], "max_iterations")

    def test_image_benchmark_uses_initial_as_before_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, image_root = self.make_dataset(temp_dir)
            train_grid_localizer_v0(dataset, image_root, Path(temp_dir) / "student", eval_mode="resubstitution")
            prompts = Path(temp_dir) / "prompts.txt"
            prompts.write_text("red cube left of blue sphere\n", encoding="utf-8")
            args = Args()
            args.student_model = str(Path(temp_dir) / "student" / "student_model.json")
            args.prompts = str(prompts)
            args.domain = "unit"
            args.output_dir = str(Path(temp_dir) / "bench")
            args.config = None
            args.generator = "mock"
            args.limit = None
            args.shard_index = 0
            args.shard_count = 1
            args.max_iterations = 1
            args.run_name = "unit_bench"
            args.keep_going = False
            run_benchmark(args)
            manifest = [json.loads(line) for line in (Path(temp_dir) / "bench" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(manifest), 1)
            self.assertTrue(manifest[0]["before_image"].endswith("decoded.ppm"))
            self.assertTrue(manifest[0]["after_image"].endswith("decoded.ppm"))

    def test_image_benchmark_prefers_raw_final_image_when_fallback_applies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts = Path(temp_dir) / "prompts.txt"
            prompts.write_text("red cube left of blue sphere\n", encoding="utf-8")
            args = Args()
            args.student_model = str(Path(temp_dir) / "student.json")
            args.prompts = str(prompts)
            args.domain = "unit"
            args.output_dir = str(Path(temp_dir) / "bench")
            args.config = None
            args.generator = "mock"
            args.limit = None
            args.shard_index = 0
            args.shard_count = 1
            args.max_iterations = 1
            args.run_name = "unit_bench"
            args.keep_going = False

            class _FakeLoop:
                def run(self, prompt, project_root="."):
                    return {
                        "initial_decoded_image": "initial.ppm",
                        "initial_grid_image": "initial-grid.ppm",
                        "final_decoded_image": "initial.ppm",
                        "final_grid_image": "initial-grid.ppm",
                        "raw_final_decoded_image": "raw-final.ppm",
                        "raw_final_grid_image": "raw-final-grid.ppm",
                        "stop_reason": "max_iterations",
                        "evaluator_calls": 2,
                        "iterations_recorded": 1,
                        "revision_records": [{"selected_token_count": 9}],
                        "artifact_root": "artifact-root",
                        "trace_path": "trace.jsonl",
                        "fallback_applied": True,
                        "final_selection_policy": "initial_on_max_error",
                    }

            with mock.patch("ascr.benchmarks.image_quality_benchmark.build_loop", return_value=_FakeLoop()):
                run_benchmark(args)
            manifest = [json.loads(line) for line in (Path(temp_dir) / "bench" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest[0]["after_image"], "raw-final.ppm")
            self.assertEqual(manifest[0]["selected_after_image"], "initial.ppm")
            self.assertTrue(manifest[0]["fallback_applied"])
            self.assertEqual(manifest[0]["final_selection_policy"], "initial_on_max_error")
            self.assertEqual(manifest[0]["selected_token_counts"], [9])

    def test_api_judge_json_helpers(self):
        payload = extract_json_object("analysis first\n```json\n{\"before_score\":0.2,\"after_score\":0.8,\"winner\":\"after\",\"reason\":\"better\"}\n```")
        judgment = normalize_judgment(payload)
        self.assertEqual(judgment["winner"], "after")
        self.assertEqual(judgment["after_score"], 0.8)

    def test_api_judge_rewrite_latest_rows_dedupes_by_sample_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            judgments = Path(temp_dir) / "judgments.jsonl"
            write_jsonl(judgments, [
                {"sample_id": "s0", "judgment": {"winner": "before"}},
                {"sample_id": "s1", "judgment": {"winner": "tie"}},
                {"sample_id": "s0", "judgment": {"winner": "after"}},
            ])
            rows = rewrite_latest_rows(judgments)
            self.assertEqual([row["sample_id"] for row in rows], ["s1", "s0"])
            self.assertEqual(rows[-1]["judgment"]["winner"], "after")

    def test_api_judge_prunes_resolved_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            errors = Path(temp_dir) / "errors.jsonl"
            write_jsonl(errors, [
                {"sample_id": "s0", "error": "old"},
                {"sample_id": "s1", "error": "still bad"},
                {"sample_id": "s0", "error": "newer"},
            ])
            rows = prune_resolved_errors(errors, {"s0"})
            self.assertEqual(rows, [{"sample_id": "s1", "error": "still bad"}])


if __name__ == "__main__":
    unittest.main()

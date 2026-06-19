import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ascr.benchmarks.lumina_native_benchmark import run_benchmark as run_lumina_native_benchmark
from ascr.cli.lumina_native_json_probe import run_probe as run_lumina_json_probe
from ascr.core.schemas import canonical_semantic_evaluation_payload
from ascr.evaluators.lumina_native import LuminaNativeEvaluator
from ascr.evaluators.lumina_native import attach_lumina_native_engine_if_available
from ascr.generators.base import _write_mock_ppm
from ascr.generators.lumina_native import LuminaNativeEngine, align_answer_generation_lengths
from ascr.training.prepare_lumina_sft_data import convert_sft_examples
from ascr.training.train_lumina_lora_smoke import _mask_codes
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


class _NaturalLanguageEngine:
    def answer_image(self, question, image_path, max_new_tokens=384):
        return "The image appears to contain a bench and a bowl."


class _ProbeEngine:
    def answer_image(self, question, image_path, max_new_tokens=384):
        if "Return JSON only" in question:
            return json.dumps({"has_error": False, "summary": "match", "regions": [], "correction_instruction": ""})
        return "natural language caption"


class _Args:
    pass


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

    def test_malformed_native_answer_abstains(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "grid.ppm"
            _write_mock_ppm(image_path, [[row + col for col in range(4)] for row in range(4)], image_size=32)
            evaluator = LuminaNativeEvaluator(engine=_NaturalLanguageEngine())
            evaluation = evaluator.evaluate("a green bench and a blue bowl", str(image_path), 0)
            self.assertFalse(evaluation.has_error)
            self.assertTrue(evaluation.should_abstain)
            self.assertIn("malformed JSON", evaluation.summary)

    def test_answer_generation_settings_are_stored_without_load(self):
        engine = LuminaNativeEngine(
            lora_path="outputs/stage2_lumina_native/lora_v2",
            answer_steps=7,
            answer_block_length=32,
            answer_temperature=0.25,
            answer_cfg_scale=1.5,
        )
        self.assertEqual(engine.lora_path, "outputs/stage2_lumina_native/lora_v2")
        self.assertEqual(engine.answer_steps, 7)
        self.assertEqual(engine.answer_block_length, 32)
        self.assertEqual(engine.answer_temperature, 0.25)
        self.assertEqual(engine.answer_cfg_scale, 1.5)

    def test_answer_generation_alignment_satisfies_lumina_blocks(self):
        gen_len, block_len, steps = align_answer_generation_lengths(384, 128, 64)
        self.assertEqual(gen_len, 384)
        self.assertEqual(block_len, 128)
        self.assertEqual(steps % (gen_len // block_len), 0)
        gen_len, block_len, steps = align_answer_generation_lengths(10, 128, 1)
        self.assertEqual(gen_len, 128)
        self.assertEqual(steps, 1)

    def test_canonical_semantic_payload_omits_runtime_fields(self):
        payload = canonical_semantic_evaluation_payload({
            "has_error": True,
            "summary": "wrong relation",
            "regions": [{
                "cells": [{"label": "B2"}],
                "reason": "wrong relation",
                "confidence": 0.8,
                "error_type": "semantic",
                "action": "reopen",
            }],
            "correction_instruction": "Fix the relation.",
            "raw": {"debug": True},
            "parser_error": "debug only",
            "should_abstain": False,
        })
        self.assertEqual(set(payload), {"has_error", "summary", "regions", "correction_instruction"})
        self.assertEqual(payload["regions"][0]["cells"], [{"label": "B2"}])

    def test_all_answer_mask_mode_masks_every_target_token(self):
        masked, labels = _mask_codes([10, 11, 12], mode="all")
        self.assertEqual(masked, [126336, 126336, 126336])
        self.assertEqual(labels, [10, 11, 12])

    def test_shared_lumina_engine_is_attached(self):
        engine = object()

        class _Generator:
            def engine(self):
                return engine

        evaluator = LuminaNativeEvaluator()
        self.assertTrue(attach_lumina_native_engine_if_available(_Generator(), evaluator))
        self.assertIs(evaluator.engine, engine)

    def test_json_probe_reports_parse_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "grid.ppm"
            _write_mock_ppm(image_path, [[row + col for col in range(4)] for row in range(4)], image_size=32)
            args = _Args()
            args.image = [str(image_path)]
            args.prompt = ["a green bench and a blue bowl"]
            args.output_dir = str(Path(temp_dir) / "probe")
            args.checkpoint_path = "models/lumina-dimoo"
            args.lora_path = "outputs/stage2_lumina_native/lora_v2"
            args.repo_path = None
            args.device = "cuda"
            args.image_size = 1024
            args.grid_size = 4
            args.max_selected_cells = 6
            args.max_new_tokens = 64
            args.answer_steps = 8
            args.answer_block_length = 16
            args.answer_temperature = 0.0
            args.answer_cfg_scale = 0.0
            args.prompt_variant = None
            summary = run_lumina_json_probe(args, engine=_ProbeEngine())
            self.assertEqual(summary["row_count"], 3)
            self.assertEqual(summary["parsed_count"], 1)
            self.assertEqual(summary["lora_path"], "outputs/stage2_lumina_native/lora_v2")
            rows = [json.loads(line) for line in (Path(temp_dir) / "probe" / "probe_rows.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("abstained_malformed_json", {row["status"] for row in rows})

    def test_lumina_native_benchmark_manifest_is_neutral(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts = Path(temp_dir) / "prompts.txt"
            prompts.write_text("red cube left of blue sphere\n", encoding="utf-8")
            args = _Args()
            args.prompts = str(prompts)
            args.domain = "unit"
            args.output_dir = str(Path(temp_dir) / "bench")
            args.config = None
            args.generator = "mock"
            args.limit = None
            args.shard_index = 0
            args.shard_count = 1
            args.max_iterations = 1
            args.run_name = "unit_native_bench"
            args.keep_going = False

            class _FakeLoop:
                def run(self, prompt, project_root="."):
                    return {
                        "initial_decoded_image": "initial.ppm",
                        "initial_grid_image": "initial-grid.ppm",
                        "final_decoded_image": "selected.ppm",
                        "final_grid_image": "selected-grid.ppm",
                        "raw_final_decoded_image": "raw-final.ppm",
                        "raw_final_grid_image": "raw-final-grid.ppm",
                        "stop_reason": "semantic_evaluator_abstained",
                        "evaluator_calls": 1,
                        "iterations_recorded": 0,
                        "revision_records": [],
                        "artifact_root": "artifact-root",
                        "trace_path": "trace.jsonl",
                        "fallback_applied": False,
                        "final_selection_policy": "last_candidate",
                    }

            with mock.patch("ascr.benchmarks.lumina_native_benchmark.build_loop", return_value=_FakeLoop()):
                run_lumina_native_benchmark(args)
            manifest = [json.loads(line) for line in (Path(temp_dir) / "bench" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest[0]["evaluator_backend"], "lumina_native_evaluator")
            self.assertNotIn("student_model", manifest[0])
            self.assertEqual(manifest[0]["before_image"], "initial.ppm")
            self.assertEqual(manifest[0]["after_image"], "raw-final.ppm")

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
            self.assertNotIn("raw", rows[0]["target_json"])
            self.assertNotIn("parser_error", rows[0]["target_json"])
            self.assertNotIn("should_abstain", rows[0]["target_json"])

    def test_convert_lumina_sft_data_skips_missing_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "grid.ppm"
            _write_mock_ppm(image, [[row + col for col in range(4)] for row in range(4)], image_size=32)
            examples = root / "sft_examples.jsonl"
            write_jsonl(examples, [
                {
                    "sample_id": "ok",
                    "prompt": "a green bench",
                    "image_path": str(image),
                    "input_text": "Return exactly one compact JSON object.",
                    "target_json": {"has_error": False, "summary": "ok", "regions": [], "correction_instruction": ""},
                },
                {
                    "sample_id": "missing",
                    "prompt": "missing image",
                    "image_path": str(root / "missing.ppm"),
                    "target_json": {"has_error": False, "summary": "ok", "regions": [], "correction_instruction": ""},
                },
            ])

            def fake_tokenizer(path):
                return {"input_ids": [1, 2, 3, 4], "height": 32, "width": 32}

            manifest = convert_sft_examples(examples, root / "lumina_format", image_tokenizer=fake_tokenizer)
            self.assertEqual(manifest["example_count"], 1)
            self.assertEqual(manifest["skipped_count"], 1)
            rows = [json.loads(line) for line in (root / "lumina_format" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["sample_id"], "ok")
            self.assertIn('"has_error":false', rows[0]["answer_text"])
            self.assertNotIn("/grp01/", rows[0]["user_image"])


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import pickle
import tempfile
import unittest

from ascr.training.prepare_lumina_sft_data import convert_sft_examples
from ascr.training.stage4_mmu_lora import (
    mmu_localization_prompt,
    prepare_mmu_sft_dataset,
    run_mmu_localization_probe,
    safe_parse_mmu_localization_payload,
)
from ascr.cli.stage4_compare_input_modes import compare_probe_summaries, write_comparison
from ascr.cli.stage4_analyze_probe_failures import analyze_probe_failures, write_outputs as write_failure_outputs
from ascr.cli.stage4_build_run_registry import build_registry, write_outputs as write_registry_outputs
from ascr.cli.stage4_summarize_curriculum import summarize_curriculum, write_outputs as write_curriculum_outputs


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def write_tokens(path, count=16):
    Path(path).write_text(json.dumps([126356 + idx for idx in range(count)]) + "\n", encoding="utf-8")


class _TokenAnswerEngine:
    def answer_vq_tokens(self, question, vq_ids, max_new_tokens=384):
        return json.dumps({
            "has_error": True,
            "corrupted_cells_2x2": ["A1"],
        })


class _ImageAnswerEngine:
    def answer_image(self, question, image_path, max_new_tokens=384):
        return json.dumps({
            "has_error": True,
            "corrupted_cells_2x2": ["A1"],
        })


class Stage4MmuLoraTests(unittest.TestCase):
    def test_prompt_requests_localization_cell_json(self):
        prompt = mmu_localization_prompt("a red cube", grid_size=2, max_selected_cells=2)
        self.assertIn("Return exactly one compact JSON object", prompt)
        self.assertIn('"corrupted_cells_2x2": string[]', prompt)
        self.assertIn("Do not put cell lists inside correction_instruction", prompt)
        self.assertIn("A1, A2, B1, B2", prompt)

    def test_prepare_mmu_sft_dataset_writes_vq_backed_canonical_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            write_tokens(tokens)
            dataset = root / "dataset.jsonl"
            write_jsonl(dataset, [{
                "sample_id": "p0000_c000",
                "prompt": "a red cube",
                "corrupted_image": "missing.ppm",
                "corrupted_vq_ids_path": str(tokens),
                "corruption_indices": [[0, 0], [0, 1]],
                "corruption_type": "block_2x2_random_replace",
                "token_grid_size": 4,
                "image_size": 64,
            }])
            manifest = prepare_mmu_sft_dataset(
                dataset,
                root / "sft",
                grid_size=2,
                max_selected_cells=4,
                eval_mode="resubstitution",
            )
            rows = [json.loads(line) for line in Path(manifest["sft_examples"]).read_text(encoding="utf-8").splitlines()]
        target = rows[0]["target_json"]
        self.assertEqual(manifest["missing_vq_ids"], 0)
        self.assertFalse(rows[0]["image_exists"])
        self.assertTrue(rows[0]["vq_ids_exists"])
        self.assertEqual(rows[0]["input_mode"], "vq_tokens")
        self.assertEqual(rows[0]["target_schema"], "localization_cells")
        self.assertEqual(set(target), {"has_error", "corrupted_cells_2x2"})
        self.assertEqual(target["corrupted_cells_2x2"], ["A1"])

    def test_lumina_sft_conversion_can_use_vq_tokens_without_image_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            write_tokens(tokens)
            examples = root / "sft_examples.jsonl"
            write_jsonl(examples, [{
                "sample_id": "p0000_c000",
                "vq_ids_path": str(tokens),
                "image_size": 64,
                "token_grid_size": 4,
                "input_text": "Locate corruption.",
                "target_json": {"has_error": False, "summary": "ok", "regions": [], "correction_instruction": ""},
            }])

            def fail_tokenizer(_path):
                raise AssertionError("image tokenizer should not be called for vq_ids_path examples")

            manifest = convert_sft_examples(
                examples,
                root / "lumina_sft",
                image_tokenizer=fail_tokenizer,
                image_size=64,
            )
            train_rows = [json.loads(line) for line in Path(manifest["train_jsonl"]).read_text(encoding="utf-8").splitlines()]
            with Path(train_rows[0]["user_image"]).open("rb") as handle:
                payload = pickle.load(handle)
        self.assertEqual(manifest["direct_vq_example_count"], 1)
        self.assertEqual(train_rows[0]["input_mode"], "vq_tokens")
        self.assertEqual(payload["height"], 64)
        self.assertEqual(payload["input_ids"][0], 126356)

    def test_lumina_sft_conversion_honors_decoded_image_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            image = root / "corrupt.ppm"
            write_tokens(tokens)
            image.write_text("fake image placeholder", encoding="utf-8")
            examples = root / "sft_examples.jsonl"
            write_jsonl(examples, [{
                "sample_id": "p0000_c000",
                "input_mode": "decoded_image",
                "vq_ids_path": str(tokens),
                "image_path": str(image),
                "image_size": 64,
                "token_grid_size": 4,
                "input_text": "Locate corruption.",
                "target_json": {"has_error": True, "corrupted_cells_2x2": ["A1"]},
            }])

            def fake_tokenizer(path):
                self.assertEqual(Path(path), image)
                return {"input_ids": [1, 2, 3, 4], "height": 32, "width": 32}

            manifest = convert_sft_examples(
                examples,
                root / "lumina_sft",
                image_tokenizer=fake_tokenizer,
                image_size=64,
            )
            train_rows = [json.loads(line) for line in Path(manifest["train_jsonl"]).read_text(encoding="utf-8").splitlines()]
            with Path(train_rows[0]["user_image"]).open("rb") as handle:
                payload = pickle.load(handle)
        self.assertEqual(manifest["direct_vq_example_count"], 0)
        self.assertEqual(manifest["image_encoded_example_count"], 1)
        self.assertEqual(train_rows[0]["input_mode"], "decoded_image")
        self.assertEqual(payload["input_ids"], [1, 2, 3, 4])

    def test_mmu_probe_scores_fake_token_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            write_tokens(tokens)
            dataset = root / "dataset.jsonl"
            write_jsonl(dataset, [{
                "sample_id": "p0000_c000",
                "prompt": "a red cube",
                "corrupted_image": "missing.ppm",
                "corrupted_vq_ids_path": str(tokens),
                "corruption_indices": [[0, 0]],
                "corruption_type": "block_2x2_random_replace",
                "token_grid_size": 4,
                "image_size": 64,
            }])
            summary = run_mmu_localization_probe(
                dataset,
                root / "probe",
                grid_size=2,
                max_selected_cells=4,
                top_k=1,
                engine=_TokenAnswerEngine(),
                use_vq_tokens=True,
            )
        self.assertEqual(summary["parse_rate"], 1.0)
        self.assertEqual(summary["input_mode"], "vq_tokens")
        self.assertEqual(summary["metrics"]["hit_any_rate"], 1.0)
        self.assertEqual(summary["metrics"]["mean_iou"], 1.0)

    def test_mmu_probe_scores_fake_image_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "corrupt.ppm"
            image.write_text("fake image placeholder", encoding="utf-8")
            dataset = root / "dataset.jsonl"
            write_jsonl(dataset, [{
                "sample_id": "p0000_c000",
                "prompt": "a red cube",
                "corrupted_image": str(image),
                "corruption_indices": [[0, 0]],
                "corruption_type": "block_2x2_random_replace",
                "token_grid_size": 4,
                "image_size": 64,
            }])
            summary = run_mmu_localization_probe(
                dataset,
                root / "probe",
                grid_size=2,
                max_selected_cells=4,
                top_k=1,
                engine=_ImageAnswerEngine(),
                input_mode="decoded_image",
            )
        self.assertEqual(summary["parse_rate"], 1.0)
        self.assertEqual(summary["input_mode"], "decoded_image")
        self.assertEqual(summary["metrics"]["hit_any_rate"], 1.0)

    def test_parser_recovers_numeric_cells_from_correction_instruction(self):
        evaluation, normalised = safe_parse_mmu_localization_payload(
            {"has_error": True, "correction_instruction": "0, 1"},
            grid_size=2,
            max_selected_cells=4,
        )
        self.assertFalse(evaluation.should_abstain)
        self.assertEqual([cell.to_label() for cell in evaluation.regions[0].cells], ["A1", "A2"])
        self.assertEqual(normalised["regions"][0]["cells"], [{"label": "A1"}, {"label": "A2"}])

    def test_compare_input_modes_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vq = root / "vq" / "summary.json"
            image = root / "image" / "summary.json"
            vq.parent.mkdir(parents=True)
            image.parent.mkdir(parents=True)
            vq.write_text(json.dumps({
                "input_mode": "vq_tokens",
                "parse_rate": 0.5,
                "malformed_count": 1,
                "call_error_count": 0,
                "metrics": {"hit_any_rate": 0.25, "mean_iou": 0.1},
            }), encoding="utf-8")
            image.write_text(json.dumps({
                "input_mode": "decoded_image",
                "parse_rate": 0.75,
                "malformed_count": 0,
                "call_error_count": 0,
                "metrics": {"hit_any_rate": 0.5, "mean_iou": 0.2},
            }), encoding="utf-8")
            comparison = compare_probe_summaries([vq, image])
            outputs = write_comparison(root / "compare", comparison)
            self.assertEqual(comparison["metrics"][1]["winner"], "decoded_image")
            self.assertTrue(Path(outputs["comparison_json"]).exists())
            self.assertTrue(Path(outputs["comparison_md"]).exists())

    def test_summarize_curriculum_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grid4 = root / "grid4" / "summary.json"
            grid8 = root / "grid8" / "summary.json"
            grid4.parent.mkdir()
            grid8.parent.mkdir()
            grid4.write_text(json.dumps({
                "grid_size": 4,
                "input_mode": "vq_tokens",
                "row_count": 2,
                "parse_rate": 1.0,
                "parsed_count": 2,
                "malformed_count": 0,
                "call_error_count": 0,
                "mean_latency_ms": 10.0,
                "metrics": {"hit_any_rate": 0.5, "mean_f1_at_k": 0.25, "mean_iou": 0.2},
            }), encoding="utf-8")
            grid8.write_text(json.dumps({
                "grid_size": 8,
                "input_mode": "vq_tokens",
                "row_count": 2,
                "parse_rate": 0.5,
                "parsed_count": 1,
                "malformed_count": 1,
                "call_error_count": 0,
                "mean_latency_ms": 12.0,
                "metrics": {"hit_any_rate": 0.0, "mean_f1_at_k": 0.0, "mean_iou": 0.0},
            }), encoding="utf-8")
            summary = summarize_curriculum([grid4, grid8], labels=["grid4", "grid8"])
            outputs = write_curriculum_outputs(root / "out", summary)
            self.assertEqual(summary["best_hit_any_rate"], 0.5)
            self.assertTrue(Path(outputs["summary_json"]).exists())
            self.assertTrue(Path(outputs["summary_md"]).exists())

    def test_analyze_probe_failures_classifies_bad_keys_and_prompt_alignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            probe_rows = root / "probe_rows.jsonl"
            sft_examples = root / "sft_examples.jsonl"
            train_jsonl = root / "train.jsonl"
            expected_prompt = mmu_localization_prompt(
                "a red cube",
                grid_size=4,
                max_selected_cells=4,
                target_schema="localization_cells",
            )
            write_jsonl(probe_rows, [
                {
                    "sample_id": "p0000_c000",
                    "status": "abstained_malformed_json",
                    "raw_text": '{"has cells":[["A_4_4x4"]]}',
                    "grid_size": 4,
                    "max_selected_cells": 4,
                    "target_schema": "localization_cells",
                    "input_mode": "vq_tokens",
                    "prompt": "a red cube",
                    "target_cells": ["A1"],
                    "predicted_cells": [],
                },
                {
                    "sample_id": "p0000_c001",
                    "status": "abstained_malformed_json",
                    "raw_text": "A1\nA2",
                    "grid_size": 4,
                    "max_selected_cells": 4,
                    "target_schema": "localization_cells",
                    "input_mode": "vq_tokens",
                    "prompt": "a red cube",
                    "target_cells": ["A2"],
                    "predicted_cells": [],
                },
            ])
            write_jsonl(sft_examples, [{
                "sample_id": "p0000_c000",
                "input_text": expected_prompt,
                "input_mode": "vq_tokens",
                "target_schema": "localization_cells",
            }])
            write_jsonl(train_jsonl, [{
                "answer_text": json.dumps({"has_error": True, "corrupted_cells_4x4": ["A1"]}, separators=(",", ":")),
            }])
            analysis = analyze_probe_failures(
                probe_rows,
                sft_examples_path=sft_examples,
                train_jsonl_path=train_jsonl,
            )
            outputs = write_failure_outputs(root / "analysis", analysis)
            self.assertEqual(analysis["classification_counts"]["wrong_key_has_cells"], 1)
            self.assertEqual(analysis["classification_counts"]["non_json_cell_label_text"], 1)
            self.assertEqual(analysis["prompt_alignment_counts"]["true"], 1)
            self.assertEqual(analysis["training_targets"]["answer_parse_error_count"], 0)
            self.assertTrue(Path(outputs["failure_summary"]).exists())
            self.assertTrue(Path(outputs["failure_rows"]).exists())

    def test_build_run_registry_collects_sft_adapter_and_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_dir = root / "grid4" / "sft"
            adapter_dir = root / "grid4" / "lora_gc"
            probe_dir = root / "grid4" / "probe_lora_gc"
            sft_dir.mkdir(parents=True)
            adapter_dir.mkdir(parents=True)
            probe_dir.mkdir(parents=True)
            (sft_dir / "manifest.json").write_text(json.dumps({
                "schema_version": "ascr.stage4.mmu_lora_sft_manifest.v1",
                "grid_size": 4,
                "input_mode": "vq_tokens",
                "target_schema": "localization_cells",
                "train_rows": 2,
                "eval_rows": 1,
            }), encoding="utf-8")
            (adapter_dir / "training_manifest.json").write_text(json.dumps({
                "schema_version": "ascr.lumina_lora_smoke.v1",
                "row_count": 2,
                "final_loss": 0.1,
                "gradient_checkpointing": True,
                "gradient_checkpointing_report": {"backend": "ascr_module_wrapper", "wrapped_module_count": 32},
            }), encoding="utf-8")
            (probe_dir / "summary.json").write_text(json.dumps({
                "schema_version": "ascr.stage4.mmu_localization_probe.summary.v1",
                "row_count": 1,
                "grid_size": 4,
                "input_mode": "vq_tokens",
                "parse_rate": 1.0,
                "lora_path": str(adapter_dir),
                "metrics": {"hit_any_rate": 0.5, "mean_iou": 0.25},
            }), encoding="utf-8")
            registry = build_registry([root])
            outputs = write_registry_outputs(root / "registry", registry)
            self.assertEqual(registry["row_count"], 3)
            kinds = {row["kind"] for row in registry["rows"]}
            self.assertEqual(kinds, {"sft_dataset", "lora_adapter", "probe_summary"})
            probe = [row for row in registry["rows"] if row["kind"] == "probe_summary"][0]
            self.assertIn("adapter_artifact_id", probe)
            self.assertTrue(Path(outputs["registry_json"]).exists())
            self.assertTrue(Path(outputs["registry_md"]).exists())


if __name__ == "__main__":
    unittest.main()

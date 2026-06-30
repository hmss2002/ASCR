import json
from pathlib import Path
import pickle
import tempfile
import unittest

from ascr.analysis.stage4_run_decision import decide_stage4_next_actions, write_next_actions
from ascr.training.prepare_lumina_sft_data import convert_sft_examples
from ascr.training.stage4_mmu_lora import (
    mmu_localization_prompt,
    prepare_mmu_sft_dataset,
    run_mmu_localization_probe,
    sample_ids_from_split_manifest,
    safe_parse_mmu_localization_payload,
)
from ascr.cli.stage4_probe_sweep import build_sweep_plan, summarize_sweep, write_summary as write_sweep_summary
from ascr.cli.stage4_batch_train import main as stage4_batch_train_main
from ascr.cli.stage4_generate_config import build_config
from ascr.cli.stage4_merge_probe_shards import merge_probe_shards
from ascr.cli.stage4_server_campaign import build_campaign_plan, write_campaign_outputs
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
        self.assertIn("Positive example", prompt)
        self.assertIn('"corrupted_cells_2x2":["A1"]', prompt)
        self.assertIn("A1, A2, B1, B2", prompt)

    def test_prompt_requests_repair_cells_json(self):
        prompt = mmu_localization_prompt(
            "a red cube",
            grid_size=8,
            max_selected_cells=8,
            target_schema="repair_cells",
        )
        self.assertIn("ASCR token-state repair cell selector", prompt)
        self.assertIn('{"cells": string[]}', prompt)
        self.assertIn('{"cells":["D4","D5"]}', prompt)
        self.assertIn('{"cells":["C3","C4","D3","D4"]}', prompt)
        self.assertIn('{"cells":[]}', prompt)
        self.assertIn('Do not output any key except "cells"', prompt)

    def test_prompt_variants_change_format_instruction(self):
        minimal = mmu_localization_prompt("a red cube", grid_size=2, prompt_variant="minimal_json")
        example = mmu_localization_prompt("a red cube", grid_size=2, prompt_variant="schema_example")
        self.assertIn("Return JSON only", minimal)
        self.assertIn("Positive example", example)
        self.assertIn('"corrupted_cells_2x2":["A1"]', example)

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

    def test_prepare_mmu_sft_dataset_writes_repair_cells_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            write_tokens(tokens, count=64 * 64)
            dataset = root / "dataset.jsonl"
            write_jsonl(dataset, [{
                "sample_id": "clean_0:pos000",
                "split_group_id": "clean_0",
                "prompt": "a red cube",
                "corrupted_vq_ids_path": str(tokens),
                "target_cells_8x8": ["A1", "A2"],
                "token_grid_size": 64,
                "image_size": 1024,
            }])
            manifest = prepare_mmu_sft_dataset(
                dataset,
                root / "sft",
                grid_size=8,
                max_selected_cells=8,
                eval_mode="resubstitution",
                target_schema="repair_cells",
            )
            rows = [json.loads(line) for line in Path(manifest["sft_examples"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["target_schema"], "repair_cells")
        self.assertEqual(rows[0]["target_json"], {"cells": ["A1", "A2"]})
        self.assertEqual(rows[0]["target_text"], '{"cells":["A1","A2"]}')

    def test_prepare_mmu_sft_dataset_keeps_clean_group_in_one_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            write_tokens(tokens, count=64 * 64)
            dataset = root / "dataset.jsonl"
            rows = []
            for group in range(5):
                rows.append({
                    "sample_id": f"clean_{group}:neg",
                    "split_group_id": f"clean_{group}",
                    "prompt": f"prompt {group}",
                    "corrupted_vq_ids_path": str(tokens),
                    "target_cells_8x8": [],
                    "token_grid_size": 64,
                })
                rows.append({
                    "sample_id": f"clean_{group}:pos000",
                    "split_group_id": f"clean_{group}",
                    "prompt": f"prompt {group}",
                    "corrupted_vq_ids_path": str(tokens),
                    "target_cells_8x8": ["A1"],
                    "token_grid_size": 64,
                })
            write_jsonl(dataset, rows)
            manifest = prepare_mmu_sft_dataset(
                dataset,
                root / "sft",
                grid_size=8,
                train_ratio=0.6,
                val_ratio=0.2,
                eval_mode="holdout",
                target_schema="repair_cells",
            )
            split_manifest = json.loads(Path(manifest["split_manifest"]).read_text(encoding="utf-8"))
        groups_by_split = {}
        for split in ("train", "val", "test"):
            indices = split_manifest[f"{split}_indices"]
            groups_by_split[split] = {rows[index]["split_group_id"] for index in indices}
        self.assertFalse(groups_by_split["train"] & groups_by_split["val"])
        self.assertFalse(groups_by_split["train"] & groups_by_split["test"])
        self.assertFalse(groups_by_split["val"] & groups_by_split["test"])

    def test_prepare_mmu_sft_dataset_writes_train_val_test_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            write_tokens(tokens)
            dataset = root / "dataset.jsonl"
            rows = []
            for index in range(10):
                rows.append({
                    "sample_id": f"p{index:04d}_c000",
                    "prompt": f"prompt {index}",
                    "corrupted_image": "missing.ppm",
                    "corrupted_vq_ids_path": str(tokens),
                    "corruption_indices": [[0, index % 4]],
                    "corruption_type": "block_2x2_random_replace",
                    "token_grid_size": 4,
                    "image_size": 64,
                })
            write_jsonl(dataset, rows)
            manifest = prepare_mmu_sft_dataset(
                dataset,
                root / "sft",
                grid_size=2,
                max_selected_cells=4,
                train_ratio=0.6,
                val_ratio=0.2,
                eval_mode="holdout",
            )
            split_manifest = json.loads(Path(manifest["split_manifest"]).read_text(encoding="utf-8"))
            train_ids = sample_ids_from_split_manifest(manifest["split_manifest"], split="train")
            val_ids = sample_ids_from_split_manifest(manifest["split_manifest"], split="val")
            test_ids = sample_ids_from_split_manifest(manifest["split_manifest"], split="test")
            eval_ids = sample_ids_from_split_manifest(manifest["split_manifest"], split="eval")
            val_examples_exists = Path(manifest["val_sft_examples"]).exists()
            test_examples_exists = Path(manifest["test_sft_examples"]).exists()
        self.assertEqual(manifest["train_rows"], 6)
        self.assertEqual(manifest["val_rows"], 2)
        self.assertEqual(manifest["test_rows"], 2)
        self.assertEqual(manifest["eval_rows"], 2)
        self.assertEqual(split_manifest["val_ratio"], 0.2)
        self.assertEqual(len(train_ids | val_ids | test_ids), 10)
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(eval_ids, test_ids)
        self.assertTrue(val_examples_exists)
        self.assertTrue(test_examples_exists)

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

    def test_mmu_probe_sample_offset_selects_disjoint_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokens = root / "corrupt.json"
            write_tokens(tokens)
            dataset = root / "dataset.jsonl"
            write_jsonl(dataset, [
                {
                    "sample_id": "first",
                    "prompt": "a red cube",
                    "corrupted_image": "missing.ppm",
                    "corrupted_vq_ids_path": str(tokens),
                    "corruption_indices": [[0, 0]],
                    "corruption_type": "block_2x2_random_replace",
                    "token_grid_size": 4,
                    "image_size": 64,
                },
                {
                    "sample_id": "second",
                    "prompt": "a blue sphere",
                    "corrupted_image": "missing.ppm",
                    "corrupted_vq_ids_path": str(tokens),
                    "corruption_indices": [[0, 2]],
                    "corruption_type": "block_2x2_random_replace",
                    "token_grid_size": 4,
                    "image_size": 64,
                },
            ])
            summary = run_mmu_localization_probe(
                dataset,
                root / "probe",
                grid_size=2,
                max_selected_cells=4,
                top_k=1,
                limit=1,
                sample_offset=1,
                engine=_TokenAnswerEngine(),
                use_vq_tokens=True,
            )
            rows = [json.loads(line) for line in (root / "probe" / "probe_rows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["sample_offset"], 1)
        self.assertEqual(summary["row_count"], 1)
        self.assertEqual(rows[0]["sample_id"], "second")

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

    def test_parser_accepts_repair_cells_payload(self):
        evaluation, normalised = safe_parse_mmu_localization_payload(
            {"cells": ["D4", "D5"]},
            grid_size=8,
            max_selected_cells=8,
            require_cells_key=True,
        )
        self.assertFalse(evaluation.should_abstain)
        self.assertEqual([cell.to_label() for cell in evaluation.regions[0].cells], ["D4", "D5"])
        self.assertTrue(normalised["has_error"])
        empty, normalised_empty = safe_parse_mmu_localization_payload(
            {"cells": []},
            grid_size=8,
            max_selected_cells=8,
            require_cells_key=True,
        )
        self.assertFalse(empty.should_abstain)
        self.assertEqual(normalised_empty["regions"], [])

    def test_parser_keeps_legacy_error_cells_compatibility(self):
        evaluation, normalised = safe_parse_mmu_localization_payload(
            {"error": True, "cells": ["D4", "D5"]},
            grid_size=8,
            max_selected_cells=8,
            require_cells_key=True,
        )
        self.assertFalse(evaluation.should_abstain)
        self.assertEqual([cell.to_label() for cell in evaluation.regions[0].cells], ["D4", "D5"])
        self.assertTrue(normalised["has_error"])
        negative, normalised_negative = safe_parse_mmu_localization_payload(
            {"error": False, "cells": ["D4"]},
            grid_size=8,
            max_selected_cells=8,
            require_cells_key=True,
        )
        self.assertFalse(negative.should_abstain)
        self.assertEqual(normalised_negative["regions"], [])

    def test_parser_rejects_missing_cells_key_in_repair_mode(self):
        with self.assertRaises(ValueError):
            safe_parse_mmu_localization_payload(
                {},
                grid_size=8,
                max_selected_cells=8,
                require_cells_key=True,
            )

    def test_parser_recovers_loose_server_cell_keys(self):
        evaluation, normalised = safe_parse_mmu_localization_payload(
            {"has cells": [["A_2_2x2"]]},
            grid_size=2,
            max_selected_cells=4,
        )
        self.assertFalse(evaluation.should_abstain)
        self.assertEqual([cell.to_label() for cell in evaluation.regions[0].cells], ["A2"])
        self.assertEqual(normalised["summary"], "Stage-4 localization cells from has cells.")

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

    def test_decide_next_recommends_gc_smoke_when_registry_is_empty(self):
        decision = decide_stage4_next_actions({"rows": []})
        commands = "\n".join(action.get("command") or "" for action in decision["actions"])
        self.assertIn("train_mmu_lora_gc_probe.sbatch", commands)

    def test_decide_next_prioritizes_format_control_after_bad_grid4_gc_probe(self):
        registry = {
            "rows": [
                {
                    "kind": "lora_adapter",
                    "artifact_id": "grid4/vq_tokens/lora_l40s_1024px_gc_adam8bit",
                    "path": "outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/lora_l40s_1024px_gc_adam8bit",
                    "image_size": 1024,
                    "epochs": 1,
                    "gradient_checkpointing": True,
                    "gradient_checkpointing_backend": "ascr_module_wrapper",
                    "wrapped_module_count": 32,
                },
                {
                    "kind": "probe_summary",
                    "artifact_id": "grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval",
                    "path": "outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval",
                    "grid_size": 4,
                    "parse_rate": 0.0,
                    "hit_any_rate": 0.0,
                },
            ]
        }
        failures = [{"classification_counts": {"wrong_key_has_cells": 4}}]
        decision = decide_stage4_next_actions(registry, failure_summaries=failures)
        titles = [action["title"] for action in decision["actions"]]
        self.assertTrue(any("schema_example" in title for title in titles))
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = write_next_actions(Path(temp_dir), decision)
            self.assertTrue(Path(outputs["next_actions_json"]).exists())
            self.assertTrue(Path(outputs["next_actions_md"]).exists())

    def test_stage4_batch_train_dry_run_uses_current_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rc = stage4_batch_train_main([
                "--dry-run",
                "--grids",
                "4",
                "--no-run-probe",
                "--output-dir",
                temp_dir,
            ])
            manifest = json.loads((Path(temp_dir) / "batch_train_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["results"][0]["kind"], "train")
        self.assertTrue(manifest["results"][0]["command"][0])

    def test_stage4_generate_config_supports_hard256(self):
        sft = build_config(4, dataset="hard256", kind="sft")
        train = build_config(4, dataset="hard256", kind="train")
        probe = build_config(8, dataset="hard256", kind="probe")
        self.assertIn("locality_hard256_v1", sft["dataset"])
        self.assertEqual(sft["train_ratio"], 0.6)
        self.assertEqual(sft["val_ratio"], 0.2)
        self.assertIn("mmu_lora_hard256_curriculum/grid4", train["output_dir"])
        self.assertIn("mmu_lora_hard256_curriculum/grid4", train["data_jsonl"])
        self.assertIn("mmu_lora_hard256_curriculum/grid4", train["val_jsonl"])
        self.assertEqual(train["checkpoint_every_epochs"], 1)
        self.assertEqual(train["early_stopping_patience"], 3)
        self.assertEqual(train["early_stopping_min_delta"], 0.0)
        self.assertEqual(train["progress_every_steps"], 25)
        self.assertTrue(train["progress_bar"])
        self.assertIn("locality_hard256_v1", probe["dataset"])
        self.assertEqual(probe["grid_size"], 8)

    def test_merge_probe_shards_recomputes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, sample_id in enumerate(["a", "b"]):
                shard = root / f"gpu_{index}"
                shard.mkdir()
                (shard / "summary.json").write_text(json.dumps({
                    "grid_size": 2,
                    "top_k": 1,
                    "input_mode": "vq_tokens",
                    "target_schema": "localization_cells",
                    "prompt_variant": "schema_example",
                    "lora_path": "adapter",
                }), encoding="utf-8")
                write_jsonl(shard / "probe_rows.jsonl", [{
                    "sample_id": sample_id,
                    "prompt": "prompt",
                    "corruption_type": "block",
                    "grid_size": 2,
                    "status": "parsed",
                    "target_cells": ["A1"],
                    "predicted_cells": ["A1"] if sample_id == "a" else [],
                    "latency_ms": 10,
                }])
            summary = merge_probe_shards([root / "gpu_0", root / "gpu_1"], root / "merged")
            summary_exists = (root / "merged" / "summary.json").exists()
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["parse_rate"], 1.0)
        self.assertEqual(summary["metrics"]["hit_any_rate"], 0.5)
        self.assertTrue(summary_exists)

    def test_probe_sweep_plan_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_sweep_plan(
                {"dataset": "dataset.jsonl", "grid_size": 4},
                root,
                prompt_variants=["default", "schema_example"],
                max_new_tokens=[128, 384],
                answer_steps=[64],
                answer_temperatures=[0.0],
                answer_cfg_scales=[0.0],
                answer_block_lengths=[128],
            )
            self.assertEqual(plan["combo_count"], 4)
            first_summary = Path(plan["combos"][0]["output_dir"]) / "summary.json"
            first_summary.parent.mkdir(parents=True)
            first_summary.write_text(json.dumps({
                "row_count": 2,
                "parse_rate": 0.5,
                "malformed_count": 1,
                "call_error_count": 0,
                "metrics": {"hit_any_rate": 0.25, "mean_f1_at_k": 0.1, "mean_iou": 0.05},
            }), encoding="utf-8")
            summary = summarize_sweep(plan)
            outputs = write_sweep_summary(root / "summary", summary)
            self.assertEqual(summary["complete_count"], 1)
            self.assertEqual(summary["best"][0]["label"], plan["combos"][0]["label"])
            self.assertTrue(Path(outputs["sweep_summary_json"]).exists())

    def test_server_campaign_plan_writes_parallel_curriculum_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_campaign_plan(grids=[4, 8, 16], output_dir=root)
            outputs = write_campaign_outputs(root, plan)
            self.assertEqual(plan["slurm_array"], "0-2")
            self.assertIn("schema_example", plan["prompt_policy"])
            self.assertIn("PROFILE=l40s_1024_gc sbatch --parsable --array=0-2", plan["primary_submit_command"])
            self.assertEqual([item["grid"] for item in plan["split_submit_commands"]], [4, 8, 16])
            shell_text = Path(outputs["campaign_shell"]).read_text(encoding="utf-8")
            self.assertIn("MODE=${MODE:-plan}", shell_text)
            self.assertTrue(Path(outputs["campaign_manifest"]).exists())


if __name__ == "__main__":
    unittest.main()

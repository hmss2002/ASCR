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
)


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
            "summary": "self-corruption in A1",
            "regions": [{
                "cells": [{"label": "A1"}],
                "reason": "localized artifact",
                "confidence": 1.0,
                "error_type": "self_corruption",
                "action": "reopen",
            }],
            "correction_instruction": "Reopen A1.",
        })


class Stage4MmuLoraTests(unittest.TestCase):
    def test_prompt_requests_canonical_reopen_json(self):
        prompt = mmu_localization_prompt("a red cube", grid_size=2, max_selected_cells=2)
        self.assertIn("Return exactly one compact JSON object", prompt)
        self.assertIn('"action": "reopen"', prompt)
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
        self.assertEqual(set(target), {"has_error", "summary", "regions", "correction_instruction"})
        self.assertEqual(target["regions"][0]["cells"], [{"label": "A1"}])

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
        self.assertEqual(summary["metrics"]["hit_any_rate"], 1.0)
        self.assertEqual(summary["metrics"]["mean_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()

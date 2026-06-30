import json
from pathlib import Path
import tempfile
import unittest

from ascr.analysis.stage3_token_repair import build_token_repair_dataset
from ascr.cli.stage3_generate_clean_tokens import generate_clean_tokens
from ascr.generators.lumina_native import IMAGE_TOKEN_OFFSET


class Stage3TokenRepairDatasetTests(unittest.TestCase):
    def test_mock_clean_token_generation_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompts = root / "prompts.txt"
            prompts.write_text("a red cube\nblue sphere beside green bowl\n", encoding="utf-8")
            summary = generate_clean_tokens(
                prompts,
                root / "clean",
                prompt_limit=2,
                token_grid_size=8,
                mock=True,
            )
            rows = [
                json.loads(line)
                for line in (root / "clean" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(len(rows), 2)
            self.assertTrue(Path(rows[0]["clean_vq_ids_path"]).exists())
            tokens = json.loads(Path(rows[0]["clean_vq_ids_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(tokens), 64)
            self.assertGreaterEqual(tokens[0], IMAGE_TOKEN_OFFSET)

    def test_build_token_repair_dataset_writes_repair_cells_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_dir = root / "clean"
            clean_dir.mkdir()
            manifest = clean_dir / "manifest.jsonl"
            clean_rows = []
            for index in range(3):
                token_path = clean_dir / f"clean_{index}.json"
                tokens = [IMAGE_TOKEN_OFFSET + ((index * 101 + value) % 8192) for value in range(64 * 64)]
                token_path.write_text(json.dumps(tokens), encoding="utf-8")
                clean_rows.append({
                    "sample_id": f"clean_{index}",
                    "prompt_index": index,
                    "prompt": f"prompt {index}",
                    "clean_vq_ids_path": str(token_path),
                    "token_grid_size": 64,
                    "image_size": 1024,
                })
            with manifest.open("w", encoding="utf-8") as handle:
                for row in clean_rows:
                    json.dump(row, handle)
                    handle.write("\n")
            dataset_manifest = build_token_repair_dataset(
                [manifest],
                root / "dataset",
                positive_rows=6,
                negative_rows=3,
                variants_per_clean=2,
                mask_sizes=[1, 2, 4, 8],
                operators=["random_replace", "local_shuffle", "neighbor_copy", "transplant"],
                action_grid_size=8,
                seed=0,
            )
            rows = [
                json.loads(line)
                for line in Path(dataset_manifest["dataset"]).read_text(encoding="utf-8").splitlines()
            ]
        positives = [row for row in rows if row["row_type"] == "positive"]
        negatives = [row for row in rows if row["row_type"] == "negative"]
        self.assertEqual(dataset_manifest["positive_rows"], 6)
        self.assertEqual(dataset_manifest["negative_rows"], 3)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["target_json"]["cells"] for row in positives))
        self.assertTrue(all(set(row["target_json"]) == {"cells"} for row in positives))
        self.assertTrue(all(row["target_json"] == {"cells": []} for row in negatives))
        for row in positives:
            for cell in row["target_json"]["cells"]:
                self.assertRegex(cell, r"^[A-H][1-8]$")

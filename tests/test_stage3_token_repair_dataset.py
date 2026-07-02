import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascr.analysis.stage3_token_repair import build_token_repair_dataset
from ascr.cli.stage3_clean_manifest_report import build_clean_manifest_report, write_outputs
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
                generation_timesteps=32,
                guidance_scale=3.0,
                temperature=0.75,
                mock=True,
            )
            rows = [
                json.loads(line)
                for line in (root / "clean" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(summary["generated_count"], 2)
            self.assertEqual(summary["skipped_existing_count"], 0)
            self.assertGreaterEqual(summary["elapsed_s"], 0.0)
            self.assertGreater(summary["rows_per_s"], 0.0)
            self.assertEqual(len(rows), 2)
            self.assertTrue(Path(rows[0]["clean_vq_ids_path"]).exists())
            tokens = json.loads(Path(rows[0]["clean_vq_ids_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(tokens), 64)
            self.assertGreaterEqual(tokens[0], IMAGE_TOKEN_OFFSET)
            self.assertEqual(rows[0]["generation_timesteps"], 32)
            self.assertEqual(rows[0]["guidance_scale"], 3.0)
            self.assertEqual(rows[0]["temperature"], 0.75)
            self.assertTrue(rows[0]["generated"])
            self.assertFalse(rows[0]["reused_existing"])
            self.assertEqual(summary["generation_timesteps"], 32)

    def test_clean_token_generation_skips_existing_without_loading_engine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompts = root / "prompts.txt"
            prompts.write_text("a red cube\n", encoding="utf-8")
            token_dir = root / "clean" / "clean_tokens"
            token_dir.mkdir(parents=True)
            token_path = token_dir / "clean_p00000_vq_ids.json"
            token_path.write_text(json.dumps([IMAGE_TOKEN_OFFSET] * 64), encoding="utf-8")
            with patch("ascr.cli.stage3_generate_clean_tokens.LuminaNativeEngine") as engine:
                summary = generate_clean_tokens(
                    prompts,
                    root / "clean",
                    prompt_limit=1,
                    token_grid_size=8,
                    mock=False,
                )
            engine.assert_not_called()
            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["generated_count"], 0)
            self.assertEqual(summary["skipped_existing_count"], 1)
            rows = [
                json.loads(line)
                for line in (root / "clean" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertFalse(rows[0]["generated"])
            self.assertTrue(rows[0]["reused_existing"])

    def test_clean_token_generation_regenerates_invalid_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompts = root / "prompts.txt"
            prompts.write_text("a red cube\n", encoding="utf-8")
            token_dir = root / "clean" / "clean_tokens"
            token_dir.mkdir(parents=True)
            token_path = token_dir / "clean_p00000_vq_ids.json"
            token_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            summary = generate_clean_tokens(
                prompts,
                root / "clean",
                prompt_limit=1,
                token_grid_size=8,
                mock=True,
            )
            tokens = json.loads(token_path.read_text(encoding="utf-8"))
            self.assertEqual(len(tokens), 64)
            self.assertEqual(summary["generated_count"], 1)
            self.assertEqual(summary["skipped_existing_count"], 0)

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

    def test_clean_manifest_report_flags_missing_and_unmanifested_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "clean"
            token_dir = output_root / "node_0000" / "gpu_00" / "clean_tokens"
            token_dir.mkdir(parents=True)
            token_path = token_dir / "clean_p00000_vq_ids.json"
            token_path.write_text(json.dumps([IMAGE_TOKEN_OFFSET] * 64), encoding="utf-8")
            unmanifested_path = token_dir / "clean_p00002_vq_ids.json"
            unmanifested_path.write_text(json.dumps([IMAGE_TOKEN_OFFSET + 1] * 64), encoding="utf-8")
            summary_path = output_root / "node_0000" / "gpu_00" / "summary.json"
            summary_path.write_text(json.dumps({
                "row_count": 2,
                "generated_count": 1,
                "skipped_existing_count": 1,
                "elapsed_s": 4.0,
            }), encoding="utf-8")
            manifest = output_root / "clean_manifest.jsonl"
            rows = [
                {
                    "sample_id": "clean_p00000",
                    "prompt_index": 0,
                    "clean_vq_ids_path": str(token_path.relative_to(root)),
                    "token_grid_size": 64,
                    "image_size": 1024,
                    "generation_timesteps": 32,
                    "guidance_scale": 4.0,
                    "temperature": 1.0,
                },
                {
                    "sample_id": "clean_p00000",
                    "prompt_index": 0,
                    "clean_vq_ids_path": str(output_root / "missing.json"),
                    "token_grid_size": 64,
                    "image_size": 1024,
                },
            ]
            with manifest.open("w", encoding="utf-8") as handle:
                for row in rows:
                    json.dump(row, handle)
                    handle.write("\n")
            report = build_clean_manifest_report(
                [manifest],
                output_root=output_root,
                project_root=root,
                min_rows=2,
            )
            self.assertFalse(report["ok"])
            self.assertIn("missing_clean_vq_files", report["failures"])
            self.assertIn("duplicate_sample_ids", report["failures"])
            self.assertIn("duplicate_prompt_indexes", report["failures"])
            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["missing_clean_vq_file_count"], 1)
            self.assertEqual(report["unmanifested_clean_vq_file_count"], 1)
            self.assertEqual(report["generation_timesteps"]["counts"], {"32": 1})
            self.assertEqual(report["summary_file_count"], 1)
            self.assertEqual(report["summary_row_count"], 2)
            self.assertEqual(report["summary_generated_count"], 1)
            self.assertEqual(report["summary_skipped_existing_count"], 1)
            self.assertEqual(report["summary_rows_per_s"], 0.5)
            outputs = write_outputs(root / "report", report)
            self.assertTrue(Path(outputs["clean_manifest_report_json"]).exists())
            self.assertTrue(Path(outputs["clean_manifest_report_md"]).exists())

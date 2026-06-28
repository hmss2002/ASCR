import json
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from ascr.analysis.stage3_self_corrupt import (
    DATASET_ROW_SCHEMA_VERSION,
    build_dataset_rows,
    build_locality_report,
    build_self_corrupt_dataset,
    read_jsonl,
)
from ascr.cli.stage3_merge_probe_shards import merge_probe_shards
from ascr.cli.stage3_locality_report import main as locality_report_main
from ascr.cli.stage3_self_corrupt_dataset import main as dataset_main
from ascr.cli.token_locality_probe import _read_prompts


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _probe_rows():
    base = {
        "schema_version": "ascr.stage3.token_locality_probe.row.v1",
        "sample_id": "p0000_c000",
        "prompt": "a red cube",
        "clean_vq_ids_path": "outputs/probe/tokens/p0000_c000_clean_vq_ids.json",
        "corrupted_vq_ids_path": "outputs/probe/tokens/p0000_c000_corrupted_vq_ids.json",
        "clean_image": "outputs/probe/images/p0000/clean.png",
        "corrupted_image": "outputs/probe/images/p0000_c000/corrupted.png",
        "corruption": {
            "corruption_type": "block_4x4_random_replace",
            "selected_indices": [[8, 8], [8, 9]],
            "selected_count": 2,
            "changed_count": 2,
            "token_grid_size": 64,
            "token_id_space": "offset_codebook",
        },
        "coarse_labels_4x4": ["A1"],
        "coarse_labels_8x8": ["B2"],
        "coarse_labels_16x16": ["C3"],
    }
    row_a = dict(base)
    row_a["metrics"] = [
        {
            "grid_size": 4,
            "token_grid_size": 64,
            "inside_energy_fraction": 0.6,
            "inside_outside_energy_ratio": 1.5,
            "center_displacement_cells": 0.5,
            "top1_cell_hit": True,
            "topk_cell_hit": True,
            "effective_radius_cells": 1,
        },
        {
            "grid_size": 8,
            "token_grid_size": 64,
            "inside_energy_fraction": 0.4,
            "inside_outside_energy_ratio": None,
            "center_displacement_cells": 1.5,
            "top1_cell_hit": True,
            "topk_cell_hit": True,
            "effective_radius_cells": 3,
        },
    ]
    row_b = dict(base)
    row_b["sample_id"] = "p0001_c000"
    row_b["metrics"] = [
        {
            "grid_size": 4,
            "token_grid_size": 64,
            "inside_energy_fraction": 0.8,
            "inside_outside_energy_ratio": 2.5,
            "center_displacement_cells": 0.7,
            "top1_cell_hit": False,
            "topk_cell_hit": True,
            "effective_radius_cells": 2,
        }
    ]
    return [row_a, row_b]


class Stage3SelfCorruptTests(unittest.TestCase):
    def test_token_locality_prompt_window_uses_offset_and_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt_file = root / "prompts.txt"
            prompt_file.write_text("p0\np1\np2\np3\np4\n", encoding="utf-8")

            class _Args:
                pass

            args = _Args()
            args.prompt = None
            args.prompt_file = str(prompt_file)
            args.limit = None
            args.prompt_offset = 2
            args.prompt_limit = 2

            prompts, window = _read_prompts({"limit": 1}, args)
        self.assertEqual(prompts, ["p2", "p3"])
        self.assertEqual(window["source_prompt_count"], 5)
        self.assertEqual(window["prompt_offset"], 2)
        self.assertEqual(window["prompt_limit"], 2)

    def test_locality_report_aggregates_by_corruption_and_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.jsonl"
            summary = root / "summary.json"
            _write_jsonl(manifest, _probe_rows())
            _write_json(summary, {"prompt_count": 2, "analysis_grids": [4, 8], "corruption_types": ["block_4x4_random_replace"]})
            report = build_locality_report(manifest, summary_path=summary)
        grid4 = [row for row in report["by_corruption_grid"] if row["grid_size"] == 4][0]
        self.assertEqual(grid4["row_count"], 2)
        self.assertAlmostEqual(grid4["mean_inside_energy_fraction"], 0.7)
        self.assertAlmostEqual(grid4["top1_hit_rate"], 0.5)
        self.assertAlmostEqual(grid4["median_effective_radius_cells"], 1.5)

    def test_dataset_rows_preserve_phase2_contract(self):
        rows = build_dataset_rows(_probe_rows()[:1], summary={"image_size": 1024, "token_grid_size": 64})
        self.assertEqual(rows[0]["schema_version"], DATASET_ROW_SCHEMA_VERSION)
        self.assertEqual(rows[0]["corruption_type"], "block_4x4_random_replace")
        self.assertEqual(rows[0]["corruption_indices"], [[8, 8], [8, 9]])
        self.assertEqual(rows[0]["coarse_labels_8x8"], ["B2"])
        self.assertEqual(rows[0]["image_size"], 1024)

    def test_dataset_builder_writes_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "probe" / "manifest.jsonl"
            summary = root / "probe" / "summary.json"
            output = root / "dataset"
            manifest.parent.mkdir()
            _write_jsonl(manifest, _probe_rows())
            _write_json(summary, {"prompt_count": 2, "image_size": 1024, "token_grid_size": 64})
            result = build_self_corrupt_dataset(manifest, output, summary_path=summary, project_root=root)
            dataset_rows = read_jsonl(output / "dataset.jsonl")
            self.assertTrue((output / "dataset_manifest.json").exists())
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(len(dataset_rows), 2)

    def test_merge_probe_shards_rewrites_offset_sample_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shard0 = root / "shard_0000"
            shard1 = root / "shard_0001"
            shard0.mkdir()
            shard1.mkdir()
            rows0 = [_probe_rows()[0]]
            rows1 = [_probe_rows()[0]]
            _write_jsonl(shard0 / "manifest.jsonl", rows0)
            _write_jsonl(shard1 / "manifest.jsonl", rows1)
            _write_json(shard0 / "summary.json", {"prompt_offset": 0, "prompt_count": 1})
            _write_json(shard1 / "summary.json", {"prompt_offset": 4, "prompt_count": 1})
            summary = merge_probe_shards(
                [shard0, shard1],
                root / "merged",
                project_root=root,
                allow_missing_paths=True,
            )
            merged_rows = read_jsonl(root / "merged" / "manifest.jsonl")
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual([row["sample_id"] for row in merged_rows], ["p0000_c000", "p0004_c000"])
        self.assertEqual(summary["prompt_indices"], [0, 4])

    def test_clis_print_help(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as report_exit:
                locality_report_main(["--help"])
            self.assertEqual(report_exit.exception.code, 0)
            with self.assertRaises(SystemExit) as dataset_exit:
                dataset_main(["--help"])
            self.assertEqual(dataset_exit.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

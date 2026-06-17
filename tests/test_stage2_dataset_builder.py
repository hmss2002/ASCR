import json
import tempfile
import unittest
from pathlib import Path

from ascr.training.build_stage2_dataset import build_dataset, main


def _trace_record(iteration, decoded_image, grid_image, has_error=True):
    return {
        "schema_version": "stage1.trace.v1",
        "iteration": iteration,
        "original_prompt": "A red cube left of a blue sphere",
        "current_prompt": "A red cube left of a blue sphere",
        "evaluation": {
            "has_error": has_error,
            "summary": "wrong relation" if has_error else "ok",
            "regions": [{"cells": [{"row": 0, "col": 1, "label": "A2"}], "reason": "wrong relation", "confidence": 0.9, "error_type": "spatial", "action": "reopen"}] if has_error else [],
            "correction_instruction": "move the red cube left",
            "raw": {
                "ofox_raw_text": "{\"has_error\": true}",
                "ofox_payload": {
                    "has_error": has_error,
                    "summary": "wrong relation" if has_error else "ok",
                    "regions": [{"cells": ["A2"], "reason": "wrong relation", "confidence": 0.9, "error_type": "spatial", "action": "reopen"}] if has_error else [],
                    "correction_instruction": "move the red cube left",
                },
            },
        },
        "reopen_mask": {
            "token_grid_size": 64,
            "selected_count": 16,
            "selected_indices": [[0, 1], [0, 2]],
            "mask": [[False] * 64 for _ in range(64)],
        },
        "artifact_paths": {
            "decoded_image": decoded_image,
            "grid_image": grid_image,
        },
        "reserved_for_stage2": {"revision_gain": None},
    }


class Stage2DatasetBuilderTests(unittest.TestCase):
    def test_builds_dataset_from_fake_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "outputs" / "run1"
            root.mkdir(parents=True)
            trace_path = root / "trace.jsonl"
            records = [
                _trace_record(0, "iterations/000/decoded.ppm", "iterations/000/grid.ppm", has_error=True),
                _trace_record(1, "iterations/001/decoded.ppm", "iterations/001/grid.ppm", has_error=False),
            ]
            trace_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({"stop_reason": "no_semantic_error"}), encoding="utf-8")

            dataset, skipped = build_dataset([root.parent.parent])

        self.assertEqual(len(skipped), 0)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["selected_4x4_cells"], ["A2"])
        self.assertEqual(dataset[0]["after_image"], "iterations/001/decoded.ppm")
        self.assertEqual(dataset[0]["stop_reason"], "no_semantic_error")

    def test_writes_skipped_report_for_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "outputs" / "run1"
            root.mkdir(parents=True)
            trace_path = root / "trace.jsonl"
            trace_path.write_text("{bad json}\n" + json.dumps(_trace_record(0, "iterations/000/decoded.ppm", "iterations/000/grid.ppm")), encoding="utf-8")
            output_path = Path(tmpdir) / "dataset.jsonl"
            skipped_path = Path(tmpdir) / "skipped.jsonl"

            rc = main([str(root.parent.parent), "--output", str(output_path), "--skipped-report", str(skipped_path)])

            skipped_lines = skipped_path.read_text(encoding="utf-8").splitlines()
            dataset_lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(rc, 0)
        self.assertEqual(len(dataset_lines), 1)
        self.assertEqual(len(skipped_lines), 1)


if __name__ == "__main__":
    unittest.main()

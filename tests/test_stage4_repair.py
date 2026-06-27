import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from ascr.cli.stage4_extract_hidden_features import main as extract_main
from ascr.cli.stage4_hidden_state_probe import main as probe_main
from ascr.cli.stage4_train_repair_head import main as train_main
from ascr.training.stage4_repair import read_json, token_positions_for_cell, train_repair_head


def _write_features(path):
    rows = []
    for idx, target in enumerate(["A1", "B2", "A1", "B2"]):
        cells = []
        for label in ["A1", "B2"]:
            if label == "A1":
                feature = [1.0, 0.0] if target == "A1" else [0.0, 1.0]
            else:
                feature = [0.0, 1.0] if target == "B2" else [1.0, 0.0]
            cells.append({"label": label, "target": label == target, "feature": feature})
        rows.append({
            "schema_version": "ascr.stage4.hidden_features.row.v1",
            "sample_id": f"p{idx:04d}",
            "prompt": f"prompt {idx}",
            "corruption_type": "block_4x4_random_replace" if idx % 2 == 0 else "local_shuffle_4x4",
            "grid_size": 2,
            "target_cells": [target],
            "hidden_layer": -1,
            "feature_dim": 2,
            "cells": cells,
        })
    Path(path).write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


class Stage4RepairTests(unittest.TestCase):
    def test_token_positions_account_for_newlines(self):
        positions = token_positions_for_cell("B2", grid_size=2, token_grid_size=4, code_start=10)
        self.assertEqual(positions, [22, 23, 27, 28])

    def test_train_repair_head_writes_model_metrics_and_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features = root / "features.jsonl"
            output = root / "repair"
            _write_features(features)
            result = train_repair_head(features, output, eval_mode="resubstitution", top_k=1, epochs=20)
            metrics = read_json(output / "metrics.json")
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["output_dir"], str(output))
        self.assertEqual(metrics["eval"]["hit_any_rate"], 1.0)
        self.assertTrue(predictions)

    def test_clis_print_help(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for cli in [probe_main, extract_main, train_main]:
                with self.assertRaises(SystemExit) as exit_info:
                    cli(["--help"])
                self.assertEqual(exit_info.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

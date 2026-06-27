import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from ascr.cli.stage3_train_selectors import main as train_selectors_main
from ascr.training.stage3_selectors import (
    read_jsonl,
    selector_examples,
    target_cells,
    train_selector_suite,
)


def _write_ppm(path, changed_cell=None, grid_size=4, image_size=16):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = []
    for row in range(image_size):
        values = []
        for col in range(image_size):
            rgb = (0, 0, 0)
            if changed_cell is not None:
                cell_row, cell_col = changed_cell
                r0 = int(cell_row * image_size / grid_size)
                r1 = int((cell_row + 1) * image_size / grid_size)
                c0 = int(cell_col * image_size / grid_size)
                c1 = int((cell_col + 1) * image_size / grid_size)
                if r0 <= row < r1 and c0 <= col < c1:
                    rgb = (255, 0, 0)
            values.extend(str(value) for value in rgb)
        pixels.append(" ".join(values))
    path.write_text(f"P3\n{image_size} {image_size}\n255\n" + "\n".join(pixels) + "\n", encoding="ascii")


def _dataset(root):
    rows = []
    specs = [
        ("p000_c000", "B2", (1, 1), "block_4x4_random_replace"),
        ("p001_c000", "B2", (1, 1), "block_4x4_random_replace"),
        ("p002_c000", "C3", (2, 2), "local_shuffle_4x4"),
        ("p003_c000", "C3", (2, 2), "local_shuffle_4x4"),
    ]
    for sample_id, label, cell, kind in specs:
        clean = root / "images" / sample_id / "clean.ppm"
        corrupt = root / "images" / sample_id / "corrupted.ppm"
        _write_ppm(clean)
        _write_ppm(corrupt, changed_cell=cell)
        rows.append({
            "schema_version": "ascr.stage3.self_corrupt_dataset.row.v1",
            "sample_id": sample_id,
            "prompt": f"prompt {sample_id}",
            "clean_image": str(clean.relative_to(root)),
            "corrupted_image": str(corrupt.relative_to(root)),
            "clean_vq_ids_path": f"tokens/{sample_id}_clean.json",
            "corrupted_vq_ids_path": f"tokens/{sample_id}_corrupt.json",
            "corruption_indices": [[cell[0] * 16, cell[1] * 16]],
            "corruption_type": kind,
            "coarse_labels_4x4": [label],
            "token_grid_size": 64,
            "image_size": 16,
        })
    dataset = root / "dataset.jsonl"
    dataset.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return dataset


class Stage3SelectorTests(unittest.TestCase):
    def test_target_cells_use_dataset_labels(self):
        row = {"coarse_labels_4x4": ["B2"], "corruption_indices": [[20, 20]], "token_grid_size": 64}
        self.assertEqual(target_cells(row, 4), ["B2"])

    def test_selector_examples_resolve_images_and_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _dataset(root)
            examples = selector_examples(dataset, grid_size=4, project_root=root)
        self.assertEqual(examples[0]["target_cells"], ["B2"])
        self.assertFalse(examples[0]["missing_corrupted_image"])

    def test_train_selector_suite_runs_all_model_light_baselines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _dataset(root)
            output = root / "selectors"
            summary = train_selector_suite(
                dataset,
                output,
                grid_sizes=[4],
                baselines=["random", "token_prior", "rgb_diff_oracle", "rgb_localizer", "prompt_rgb_localizer"],
                eval_mode="resubstitution",
                top_k=1,
                project_root=root,
                epochs=5,
            )
            oracle = [row for row in summary["results"] if row["baseline"] == "rgb_diff_oracle"][0]
            predictions = read_jsonl(output / "grid4" / "rgb_diff_oracle" / "predictions.jsonl")
            self.assertTrue((output / "summary.json").exists())
        self.assertEqual(len(summary["results"]), 5)
        self.assertEqual(oracle["hit_any_rate"], 1.0)
        self.assertTrue(predictions)

    def test_cli_help(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                train_selectors_main(["--help"])
        self.assertEqual(exit_info.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

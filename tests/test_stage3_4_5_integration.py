import json
from pathlib import Path
import tempfile
import unittest

from ascr.analysis.stage4_failure_router import route_stage4_failure
from ascr.analysis.stage6_transfer_metrics import summarize_transfer_gap
from ascr.cli.stage3_sample_prompts import sample_prompts
from ascr.cli.stage5_self_corrupt_loop import run_stage5_loop
from ascr.cli.stage5_compare_loop_results import summarize_loop_manifest
from ascr.selectors.mmu_localizer_selector import MMULocalizerSelector


class Stage345IntegrationTests(unittest.TestCase):
    def test_mmu_localizer_selector_projects_cells_to_token_mask(self):
        selector = MMULocalizerSelector('{"has_error":true,"corrupted_cells_4x4":["D3"]}', grid_size=4, token_grid_size=64)
        mask = selector.to_token_mask()
        self.assertEqual(mask.count(), 16 * 16)
        self.assertIn((48, 32), mask.selected_indices())
        self.assertIn((63, 47), mask.selected_indices())

    def test_stage5_mock_loop_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = run_stage5_loop(
                "a green bench and a blue bowl",
                Path(temp_dir) / "loop",
                config={"mock": True, "grid_size": 4, "token_grid_size": 64},
                mock=True,
            )
            self.assertTrue(Path(trace["repaired_image"]).exists())
            self.assertGreater(trace["mask_stats"]["selected_token_count"], 0)
            summary = summarize_loop_manifest([{
                "status": "ok",
                "mask_stats": trace["mask_stats"],
                "lora_cells": trace["lora_cells"],
                "reopen_changed": trace["reopen_changed"],
            }])
            self.assertEqual(summary["mask_nonempty_rate"], 1.0)

    def test_prompt_sampler_removes_holdout_and_stratifies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            holdout = root / "holdout.txt"
            source.write_text("red cube\nblue sphere left of green bowl\nlong prompt with many objects and spatial relations left right above below\n", encoding="utf-8")
            holdout.write_text("red cube\n", encoding="utf-8")
            selected, candidates = sample_prompts([source], 2, holdout=[holdout], seed=0)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(len(selected), 2)
            self.assertNotIn("red cube", {row["prompt"] for row in selected})

    def test_failure_router_and_transfer_metrics(self):
        route = route_stage4_failure(parse_rate=0.6, hit_any_rate=0.0)
        self.assertEqual(route["route"], "more_data_or_capacity")
        metrics = summarize_transfer_gap(
            [{"target_cells": ["A1"], "lora_cells": ["A1"]}],
            [{"lora_cells": ["B2"], "status": "parsed"}],
        )
        self.assertEqual(metrics["synthetic_hit_any_rate"], 1.0)
        self.assertEqual(metrics["transfer_nonempty_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()


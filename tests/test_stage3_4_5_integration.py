import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascr.analysis.stage4_failure_router import route_stage4_failure
from ascr.analysis.stage6_transfer_metrics import summarize_transfer_gap
from ascr.cli.stage3_sample_prompts import sample_prompts
from ascr.cli.stage5_self_corrupt_loop import run_stage5_loop
from ascr.cli.stage5_compare_loop_results import summarize_loop_manifest
from ascr.selectors.mmu_localizer_selector import MMULocalizerSelector


class _CountingStage5Engine:
    token_grid_size = 64
    image_size = 1024

    def __init__(self):
        self.lora_path = None
        self.generated = 0
        self.answered = 0
        self.reopened = 0

    def generate(self, prompt, seed=0):
        self.generated += 1
        return [int((idx + seed) % 8192) for idx in range(self.token_grid_size * self.token_grid_size)]

    def reopen(self, baseline_vq_ids, selected_indices, prompt, seed=0):
        self.reopened += 1
        repaired = list(baseline_vq_ids)
        for row, col in selected_indices:
            index = int(row) * self.token_grid_size + int(col)
            if 0 <= index < len(repaired):
                repaired[index] = (int(repaired[index]) + 1) % 8192
        return repaired

    def decode_to(self, vq_ids, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("P3\n1 1\n255\n1 2 3\n", encoding="ascii")
        return str(output_path)

    def answer_vq_tokens(self, question, vq_ids, max_new_tokens=384):
        self.answered += 1
        return '{"has_error":true,"corrupted_cells_4x4":["A1"]}'


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

    def test_stage5_share_engine_reuses_one_lumina_instance_and_attaches_lora_lazily(self):
        created = []

        def build_engine(config, lora_path=None, mock=False):
            engine = _CountingStage5Engine()
            if lora_path:
                engine.lora_path = str(lora_path)
            created.append(engine)
            return engine

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("ascr.cli.stage5_self_corrupt_loop._engine", side_effect=build_engine):
                trace = run_stage5_loop(
                    "a green bench and a blue bowl",
                    Path(temp_dir) / "loop",
                    config={
                        "share_engine": True,
                        "grid_size": 4,
                        "token_grid_size": 64,
                        "lora_path": "outputs/adapters/grid4",
                    },
                )
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].generated, 1)
        self.assertEqual(created[0].answered, 1)
        self.assertEqual(created[0].reopened, 1)
        self.assertEqual(created[0].lora_path, "outputs/adapters/grid4")
        self.assertTrue(trace["share_engine"])
        self.assertEqual(trace["answer_method"], "answer_vq_tokens")

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

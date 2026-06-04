import sys
import tempfile
import unittest
from pathlib import Path

from ascr.evaluators.mmada_self_coarse import MMaDASelfCoarseEvaluator, _parse_letter_cells
from ascr.evaluators.registry import build_evaluator
from ascr.generators.mmada import MMaDAAdapter
from ascr.revision.selector import GridSemanticReopeningSelector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mmada_self_wiring import _FakeEngine  # noqa: E402


class MMaDACoarseRegistryTests(unittest.TestCase):
    def test_registry_builds_coarse_evaluator(self):
        config = {
            "coarse_grid_size": 4,
            "token_grid_size": 32,
            "evaluator": {"name": "mmada_self_coarse", "grid_size": 4, "max_selected_cells": 6},
        }
        evaluator = build_evaluator("mmada_self_coarse", config)
        self.assertIsInstance(evaluator, MMaDASelfCoarseEvaluator)
        self.assertEqual(evaluator.grid_size, 4)
        self.assertEqual(evaluator.token_grid_size, 32)

    def test_use_engine_shares_and_rejects(self):
        generator = MMaDAAdapter()
        evaluator = MMaDASelfCoarseEvaluator()
        engine = generator._engine()
        self.assertTrue(evaluator.use_engine(engine))
        self.assertIs(evaluator._engine_instance(), engine)
        self.assertIs(evaluator._engine_instance(), generator._engine())
        self.assertFalse(evaluator.use_engine(object()))


class MMaDACoarseParsingTests(unittest.TestCase):
    def test_parse_letter_cells(self):
        cells = _parse_letter_cells("The problems are in A1, B2 and D4.", 4)
        self.assertEqual(cells, [[0, 0], [1, 1], [3, 3]])

    def test_parse_letter_cells_respects_bounds(self):
        cells = _parse_letter_cells("Z9 is out of range but C3 is fine", 4)
        self.assertEqual(cells, [[2, 2]])

    def test_evaluate_match_returns_no_error(self):
        evaluator = MMaDASelfCoarseEvaluator()
        evaluator._engine = _FakeEngine(["Yes, it matches."])
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            img = Path(tmp) / "grid.png"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(img)
            result = evaluator.evaluate("a blue sphere", str(img), 0)
        self.assertFalse(result.has_error)

    def test_evaluate_localizes_coarse_cells(self):
        evaluator = MMaDASelfCoarseEvaluator(max_selected_cells=6)
        evaluator._engine = _FakeEngine(["No, wrong layout.", "Issues in A1 and B2."])
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            img = Path(tmp) / "grid.png"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(img)
            result = evaluator.evaluate("a scene", str(img), 0)
        self.assertTrue(result.has_error)
        cells = {(c.row, c.col) for region in result.regions for c in region.cells}
        self.assertEqual(cells, {(0, 0), (1, 1)})

    def test_confidence_fallback_pools_into_coarse_cells(self):
        evaluator = MMaDASelfCoarseEvaluator(max_selected_cells=2, confidence_fallback=True, confidence_fallback_cells=1)
        # localization text yields no cells -> coarse confidence fallback fires.
        evaluator._engine = _FakeEngine(["No, broken.", "hmm"])
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            Image.new("RGB", (64, 64), (0, 0, 0)).save(Path(tmp) / "decoded.ppm")
            grid = Path(tmp) / "grid.ppm"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(grid)
            result = evaluator.evaluate("a scene", str(grid), 0)
        self.assertTrue(result.has_error)
        cells = {(c.row, c.col) for region in result.regions for c in region.cells}
        # _FakeEngine.token_confidence makes indices 0..4 (top-left coarse cell) lowest.
        self.assertEqual(cells, {(0, 0)})

    def test_confidence_fallback_disabled_abstains(self):
        evaluator = MMaDASelfCoarseEvaluator(confidence_fallback=False)
        evaluator._engine = _FakeEngine(["No, broken.", "hmm"])
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            Image.new("RGB", (64, 64), (0, 0, 0)).save(Path(tmp) / "decoded.ppm")
            grid = Path(tmp) / "grid.ppm"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(grid)
            result = evaluator.evaluate("a scene", str(grid), 0)
        self.assertFalse(result.has_error)


class MMaDACoarseSelectorTests(unittest.TestCase):
    def test_coarse_cell_projects_with_dilation(self):
        evaluator = MMaDASelfCoarseEvaluator(max_selected_cells=6)
        evaluator._engine = _FakeEngine(["No.", "A1"])
        selector = GridSemanticReopeningSelector(coarse_grid_size=4, token_grid_size=32, dilation=1)
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            img = Path(tmp) / "grid.png"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(img)
            evaluation = evaluator.evaluate("a scene", str(img), 0)
        mask = selector.select(evaluation)
        # A1 = coarse (0,0) -> 8x8 token block; dilation=1 adds a one-token ring (9x9 corner).
        self.assertGreaterEqual(mask.count(), 64)
        self.assertTrue(mask.any())


class MMaDACoarseLoopTests(unittest.TestCase):
    def test_full_coarse_loop_with_shared_fake_engine(self):
        from ascr.core.loop import ASCRLoop, run_config_from_mapping

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "run_name": "mmada_coarse_test",
                "output_dir": str(Path(tmp) / "out"),
                "max_iterations": 2,
                "image_size": 512,
                "coarse_grid_size": 4,
                "token_grid_size": 32,
                "dilation": 1,
                "return_initial_on_max_error": True,
            }
            generator = MMaDAAdapter(token_grid_size=32, image_size=512)
            evaluator = MMaDASelfCoarseEvaluator(grid_size=4, token_grid_size=32)
            shared = _FakeEngine(["Yes, it matches the prompt."])
            generator._native_engine = shared
            evaluator._engine = shared
            selector = GridSemanticReopeningSelector(coarse_grid_size=4, token_grid_size=32, dilation=1)
            run_config = run_config_from_mapping(config)
            loop = ASCRLoop(generator, evaluator, selector, run_config)
            summary = loop.run("a red cube", project_root=Path(tmp))
            self.assertEqual(summary["stop_reason"], "no_semantic_error")
            self.assertTrue(Path(summary["final_decoded_image"]).exists())


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from ascr.evaluators.mmada_self import MMaDASelfEvaluator, _looks_like_match, _parse_cells_from_text
from ascr.evaluators.registry import build_evaluator
from ascr.generators.lumina_dimoo import LuminaAdapter
from ascr.generators.mmada import MMaDAAdapter
from ascr.generators.mmada_native import MMaDANativeEngine
from ascr.generators.registry import build_generator


class _FakeEngine:
    """Stand-in for MMaDANativeEngine that needs no torch / 8B weights."""

    def __init__(self, answers, token_grid_size=32):
        self._answers = list(answers)
        self._calls = 0
        self.token_grid_size = token_grid_size
        self.mask_token_id = 126336
        self.num_vq_tokens = token_grid_size * token_grid_size

    def answer_image(self, question, image_path, max_new_tokens=256):
        answer = self._answers[min(self._calls, len(self._answers) - 1)]
        self._calls += 1
        return answer

    def run_confidence_block(self, prompt, model_tokens=None, steps=None, seed=None):
        tokens = list(range(self.num_vq_tokens))
        return {
            "model_tokens": tokens,
            "decoded_tokens": tokens,
            "confidence": [],
            "confidence_mask": [],
            "confidence_steps": steps or 1,
            "step_records": [],
            "mask_token_id": self.mask_token_id,
        }

    def decode_tokens(self, decoded_tokens, output_path):
        from PIL import Image
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), (120, 90, 60)).save(output_path)
        return output_path

    def encode_image(self, image_path):
        return list(range(self.num_vq_tokens))

    def token_confidence(self, prompt, model_tokens):
        confidences = [0.9] * self.num_vq_tokens
        for i in range(5):
            confidences[i] = 0.01 * (i + 1)
        return confidences

    def force_mask(self, model_tokens, token_mask):
        next_tokens = list(model_tokens)
        for row, col in token_mask.selected_indices():
            index = int(row) * self.token_grid_size + int(col)
            if 0 <= index < len(next_tokens):
                next_tokens[index] = self.mask_token_id
        return next_tokens


class MMaDARegistryTests(unittest.TestCase):
    def test_registry_builds_lumina_generator_without_load(self):
        config = {
            "token_grid_size": 64,
            "image_size": 1024,
            "generator": {"name": "lumina", "checkpoint_path": "models/lumina-dimoo"},
        }
        generator = build_generator("lumina", config)
        self.assertIsInstance(generator, LuminaAdapter)
        self.assertEqual(generator.token_grid_size, 64)
        self.assertEqual(generator.image_size, 1024)
        self.assertIsNone(generator._engine)

    def test_registry_builds_generator_without_load(self):
        config = {"token_grid_size": 32, "image_size": 512, "generator": {"name": "mmada"}}
        generator = build_generator("mmada", config)
        self.assertIsInstance(generator, MMaDAAdapter)
        self.assertEqual(generator.token_grid_size, 32)
        self.assertIsNone(generator._native_engine)

    def test_registry_builds_self_evaluator(self):
        config = {
            "image_size": 512,
            "select_grid_size": 32,
            "evaluator": {"name": "mmada_self", "grid_size": 32, "max_selected_cells": 48},
        }
        evaluator = build_evaluator("mmada_self", config)
        self.assertIsInstance(evaluator, MMaDASelfEvaluator)
        self.assertEqual(evaluator.grid_size, 32)
        self.assertEqual(evaluator.max_selected_cells, 48)


class MMaDASharedEngineTests(unittest.TestCase):
    def test_attach_engine_shares_single_instance(self):
        generator = MMaDAAdapter()
        evaluator = MMaDASelfEvaluator()
        engine = generator._engine()
        self.assertIsInstance(engine, MMaDANativeEngine)
        self.assertTrue(evaluator.attach_engine(engine))
        self.assertIs(evaluator._engine_instance(), engine)
        self.assertIs(evaluator._engine_instance(), generator._engine())

    def test_attach_rejects_non_engine(self):
        evaluator = MMaDASelfEvaluator()
        self.assertFalse(evaluator.attach_engine(object()))


class MMaDASelfEvaluatorParsingTests(unittest.TestCase):
    def test_looks_like_match(self):
        self.assertTrue(_looks_like_match("Yes, it matches the prompt."))
        self.assertFalse(_looks_like_match("No, the cube is red not blue."))
        self.assertIsNone(_looks_like_match(""))

    def test_parse_cells_from_text(self):
        cells = _parse_cells_from_text("The problems are at R3C4 and R5C6.", 32, 64)
        self.assertEqual(cells, [[3, 4], [5, 6]])

    def test_parse_cells_respects_grid_bounds_and_cap(self):
        cells = _parse_cells_from_text("R0C0 R40C40 R1C1 R2C2", 32, 2)
        self.assertEqual(cells, [[0, 0], [1, 1]])

    def test_evaluate_match_returns_no_error(self):
        evaluator = MMaDASelfEvaluator()
        evaluator._engine = _FakeEngine(["Yes, the image fully matches the prompt."])
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "grid.png"
            from PIL import Image
            Image.new("RGB", (64, 64), (0, 0, 0)).save(image)
            result = evaluator.evaluate("a blue sphere", str(image), 0)
        self.assertFalse(result.has_error)

    def test_evaluate_localizes_token_cells(self):
        evaluator = MMaDASelfEvaluator(max_selected_cells=64)
        evaluator._engine = _FakeEngine([
            "No, the sphere is the wrong color.",
            "The wrong cells are R3C4 and R5C6.",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "grid.png"
            from PIL import Image
            Image.new("RGB", (64, 64), (0, 0, 0)).save(image)
            result = evaluator.evaluate("a blue sphere", str(image), 0)
        self.assertTrue(result.has_error)
        cells = {(c.row, c.col) for region in result.regions for c in region.cells}
        self.assertEqual(cells, {(3, 4), (5, 6)})

    def test_evaluate_caps_selected_cells(self):
        evaluator = MMaDASelfEvaluator(max_selected_cells=2)
        coords = " ".join(f"R{i}C{i}" for i in range(10))
        evaluator._engine = _FakeEngine(["No, several issues.", coords])
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "grid.png"
            from PIL import Image
            Image.new("RGB", (64, 64), (0, 0, 0)).save(image)
            result = evaluator.evaluate("a scene", str(image), 0)
        total = sum(len(region.cells) for region in result.regions)
        self.assertLessEqual(total, 2)

    def test_confidence_fallback_when_localization_fails(self):
        evaluator = MMaDASelfEvaluator(max_selected_cells=64, confidence_fallback=True, confidence_fallback_cells=5)
        evaluator._engine = _FakeEngine(["No, the layout is wrong.", "1"])
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            Image.new("RGB", (64, 64), (0, 0, 0)).save(Path(tmp) / "decoded.ppm")
            grid = Path(tmp) / "grid.ppm"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(grid)
            result = evaluator.evaluate("a scene", str(grid), 0)
        self.assertTrue(result.has_error)
        cells = {(c.row, c.col) for region in result.regions for c in region.cells}
        self.assertEqual(cells, {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)})

    def test_confidence_fallback_disabled_abstains(self):
        evaluator = MMaDASelfEvaluator(max_selected_cells=64, confidence_fallback=False)
        evaluator._engine = _FakeEngine(["No, the layout is wrong.", "1"])
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            Image.new("RGB", (64, 64), (0, 0, 0)).save(Path(tmp) / "decoded.ppm")
            grid = Path(tmp) / "grid.ppm"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(grid)
            result = evaluator.evaluate("a scene", str(grid), 0)
        self.assertFalse(result.has_error)


class MMaDASelfLoopTests(unittest.TestCase):
    def test_full_direct_loop_with_shared_fake_engine(self):
        from ascr.core.loop import run_config_from_mapping
        from ascr.core.loop_direct import DirectTokenReopenLoop
        from ascr.revision.selector import DirectTokenReopeningSelector

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "run_name": "mmada_self_test",
                "output_dir": str(Path(tmp) / "out"),
                "max_iterations": 2,
                "image_size": 512,
                "coarse_grid_size": 4,
                "token_grid_size": 32,
                "select_grid_size": 32,
                "return_initial_on_max_error": True,
            }
            generator = MMaDAAdapter()
            evaluator = MMaDASelfEvaluator()
            shared = _FakeEngine(["Yes, the image matches the prompt."])
            generator._native_engine = shared
            evaluator._engine = shared
            selector = DirectTokenReopeningSelector(token_grid_size=32, dilation=0)
            run_config = run_config_from_mapping(config)
            loop = DirectTokenReopenLoop(generator, evaluator, selector, run_config, label_step=4)
            summary = loop.run("a red cube", project_root=Path(tmp))
            self.assertEqual(summary["stage1_variant"], "direct_token")
            self.assertEqual(summary["stop_reason"], "no_semantic_error")
            self.assertTrue(Path(summary["final_decoded_image"]).exists())


if __name__ == "__main__":
    unittest.main()

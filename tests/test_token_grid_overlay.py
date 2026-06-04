import tempfile
import unittest
from pathlib import Path

from ascr.generators.base import _write_mock_ppm
from ascr.grids.overlay import create_token_grid_overlay


class TokenGridOverlayTests(unittest.TestCase):
    def test_overlay_is_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "decoded.ppm"
            grid = [[(row * 4 + col) % 127 for col in range(8)] for row in range(8)]
            _write_mock_ppm(source, grid, image_size=64)
            output = Path(temp_dir) / "grid.ppm"
            result = create_token_grid_overlay(source, output, image_size=64, token_grid_size=32, label_step=4)
            self.assertTrue(Path(result).exists())
            self.assertGreater(Path(result).stat().st_size, 0)

    def test_overlay_handles_missing_source_via_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "grid.ppm"
            result = create_token_grid_overlay(Path(temp_dir) / "missing.ppm", output, image_size=32, token_grid_size=32, label_step=8)
            self.assertTrue(Path(result).exists())
            header = Path(result).read_text(encoding="ascii").splitlines()[0]
            self.assertEqual(header, "P3")


if __name__ == "__main__":
    unittest.main()

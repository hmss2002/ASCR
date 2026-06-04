import unittest

from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.revision.selector import DirectTokenReopeningSelector


def _evaluation(cells):
    regions = [RegionSelection(cells=[GridCell.from_any(cell, 32) for cell in cells], reason="x", confidence=1.0, error_type="semantic", action="reopen")]
    return SemanticEvaluation(True, summary="err", regions=regions)


class DirectTokenSelectorTests(unittest.TestCase):
    def test_token_cells_map_one_to_one(self):
        selector = DirectTokenReopeningSelector(token_grid_size=32, dilation=0)
        mask = selector.select(_evaluation(["R0C0", "R31C31", "R5C7"]))
        self.assertEqual(mask.count(), 3)
        self.assertTrue(mask.mask[0][0])
        self.assertTrue(mask.mask[31][31])
        self.assertTrue(mask.mask[5][7])
        self.assertFalse(mask.mask[6][7])

    def test_no_projection_block_expansion(self):
        selector = DirectTokenReopeningSelector(token_grid_size=32, dilation=0)
        mask = selector.select(_evaluation(["R10C10"]))
        self.assertEqual(mask.count(), 1)

    def test_dilation_expands_neighbors(self):
        selector = DirectTokenReopeningSelector(token_grid_size=32, dilation=1)
        mask = selector.select(_evaluation(["R10C10"]))
        self.assertEqual(mask.count(), 9)
        self.assertTrue(mask.mask[9][9])
        self.assertTrue(mask.mask[11][11])

    def test_corner_dilation_clipped(self):
        selector = DirectTokenReopeningSelector(token_grid_size=32, dilation=1)
        mask = selector.select(_evaluation(["R0C0"]))
        self.assertEqual(mask.count(), 4)

    def test_empty_evaluation_returns_empty_mask(self):
        selector = DirectTokenReopeningSelector(token_grid_size=32, dilation=0)
        mask = selector.select(SemanticEvaluation(False, summary="ok"))
        self.assertEqual(mask.count(), 0)
        self.assertEqual(mask.token_grid_size, 32)

    def test_intermediate_select_grid_scales_to_token_grid(self):
        selector = DirectTokenReopeningSelector(token_grid_size=32, select_grid_size=16, dilation=0)
        mask = selector.select(_evaluation_at(["R0C0"], 16))
        self.assertEqual(mask.count(), 4)
        self.assertTrue(mask.mask[0][0])
        self.assertTrue(mask.mask[1][1])
        self.assertFalse(mask.mask[2][2])

    def test_non_divisible_select_grid_raises(self):
        with self.assertRaises(ValueError):
            DirectTokenReopeningSelector(token_grid_size=32, select_grid_size=5)


def _evaluation_at(cells, grid_size):
    regions = [RegionSelection(cells=[GridCell.from_any(cell, grid_size) for cell in cells], reason="x", confidence=1.0, error_type="semantic", action="reopen")]
    return SemanticEvaluation(True, summary="err", regions=regions)


if __name__ == "__main__":
    unittest.main()

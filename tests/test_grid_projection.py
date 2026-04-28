import unittest

from ascr.grids.projection import project_cells_to_token_mask


class GridProjectionTests(unittest.TestCase):
    def test_single_cell_without_dilation_maps_to_four_by_four_block(self):
        mask = project_cells_to_token_mask(["A1"], coarse_grid_size=4, token_grid_size=16, dilation=0)
        self.assertEqual(mask.count(), 16)
        self.assertTrue(mask.mask[0][0])
        self.assertTrue(mask.mask[3][3])
        self.assertFalse(mask.mask[4][4])

    def test_single_middle_cell_with_dilation_expands_boundary(self):
        mask = project_cells_to_token_mask(["B2"], coarse_grid_size=4, token_grid_size=16, dilation=1)
        self.assertEqual(mask.count(), 36)
        self.assertTrue(mask.mask[3][3])
        self.assertTrue(mask.mask[8][8])
        self.assertFalse(mask.mask[2][2])

    def test_corner_dilation_is_clipped(self):
        mask = project_cells_to_token_mask(["A1"], coarse_grid_size=4, token_grid_size=16, dilation=1)
        self.assertEqual(mask.count(), 25)
        self.assertTrue(mask.mask[0][0])
        self.assertTrue(mask.mask[4][4])
        self.assertFalse(mask.mask[5][5])

    def test_duplicates_do_not_change_count(self):
        mask = project_cells_to_token_mask(["C3", {"row": 2, "col": 2}], coarse_grid_size=4, token_grid_size=16, dilation=0)
        self.assertEqual(mask.count(), 16)


if __name__ == "__main__":
    unittest.main()

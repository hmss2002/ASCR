import unittest

from ascr.analysis.token_locality import diff_energy_grid, summarise_locality


class TokenLocalityTests(unittest.TestCase):
    def test_diff_energy_grid_localises_changed_pixels(self):
        clean = [[(0.0, 0.0, 0.0) for _col in range(4)] for _row in range(4)]
        corrupted = [[(0.0, 0.0, 0.0) for _col in range(4)] for _row in range(4)]
        corrupted[2][2] = (1.0, 1.0, 1.0)
        energy = diff_energy_grid(clean, corrupted, grid_size=2)
        self.assertEqual(energy[0][0], 0.0)
        self.assertGreater(energy[1][1], 0.0)

    def test_summarise_locality_reports_top_hit(self):
        energy = [[0.0, 0.0], [0.0, 10.0]]
        summary = summarise_locality(energy, selected_indices=[(3, 3)], token_grid_size=4)
        self.assertTrue(summary["top1_cell_hit"])
        self.assertEqual(summary["selected_cells"], ["B2"])
        self.assertEqual(summary["inside_energy_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()

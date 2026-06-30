import unittest

from ascr.corruption.vq_corruptor import (
    OFFSET_TOKEN_SPACE,
    corrupt_vq_ids,
    corrupt_vq_ids_with_operator,
    infer_token_id_space,
    token_indices_to_cell_labels,
)
from ascr.generators.lumina_native import IMAGE_TOKEN_OFFSET


class VQCorruptorTests(unittest.TestCase):
    def test_random_replace_changes_only_selected_token(self):
        clean = [IMAGE_TOKEN_OFFSET + value for value in range(16)]
        result = corrupt_vq_ids(clean, 4, "single_random_replace", seed=3, selected_indices=[(1, 2)])
        changed = [index for index, (before, after) in enumerate(zip(clean, result.corrupted_vq_ids)) if before != after]
        self.assertEqual(changed, [6])
        self.assertEqual(result.selected_indices, [(1, 2)])
        self.assertEqual(result.token_id_space, OFFSET_TOKEN_SPACE)

    def test_local_shuffle_keeps_values_inside_selected_block(self):
        clean = list(range(16))
        result = corrupt_vq_ids(clean, 4, "local_shuffle_2x2", seed=5, selected_indices=[(0, 0), (0, 1), (1, 0), (1, 1)])
        selected_positions = [0, 1, 4, 5]
        self.assertEqual(sorted(result.corrupted_vq_ids[pos] for pos in selected_positions), [0, 1, 4, 5])
        outside = [pos for pos in range(16) if pos not in selected_positions]
        self.assertEqual([result.corrupted_vq_ids[pos] for pos in outside], [clean[pos] for pos in outside])

    def test_infer_offset_space(self):
        self.assertEqual(infer_token_id_space([IMAGE_TOKEN_OFFSET, IMAGE_TOKEN_OFFSET + 7]), OFFSET_TOKEN_SPACE)

    def test_token_indices_project_to_grid_labels(self):
        labels = token_indices_to_cell_labels([(8, 9), (9, 9)], token_grid_size=16, grid_size=4)
        self.assertEqual(labels, ["C3"])
        direct_labels = token_indices_to_cell_labels([(27, 2)], token_grid_size=64, grid_size=64)
        self.assertEqual(direct_labels, ["R27C2"])

    def test_operator_api_supports_neighbor_copy_and_transplant(self):
        clean = [IMAGE_TOKEN_OFFSET + value for value in range(64)]
        selected = [(2, 2), (2, 3), (3, 2), (3, 3)]
        neighbor = corrupt_vq_ids_with_operator(
            clean,
            token_grid_size=8,
            mask_size=2,
            operator="neighbor_copy",
            seed=0,
            selected_indices=selected,
        )
        transplant = corrupt_vq_ids_with_operator(
            clean,
            token_grid_size=8,
            mask_size=2,
            operator="transplant",
            seed=1,
            selected_indices=selected,
        )
        self.assertEqual(neighbor.operator, "neighbor_copy")
        self.assertEqual(neighbor.mask_size, 2)
        self.assertEqual(neighbor.source_mode, "same_image_neighbor")
        self.assertEqual(neighbor.selected_indices, selected)
        self.assertGreater(neighbor.changed_count, 0)
        self.assertEqual(transplant.operator, "transplant")
        self.assertEqual(transplant.source_mode, "same_image_far")
        self.assertGreater(transplant.changed_count, 0)

    def test_local_shuffle_rejects_single_token_mask(self):
        clean = list(range(16))
        with self.assertRaises(ValueError):
            corrupt_vq_ids_with_operator(clean, 4, 1, "local_shuffle", seed=0)


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from ascr.core.schemas import GridCell, parse_semantic_evaluation, safe_parse_semantic_evaluation


class SchemaParserTests(unittest.TestCase):
    def test_grid_cell_labels(self):
        self.assertEqual(GridCell.from_any("A1").to_dict()["row"], 0)
        self.assertEqual(GridCell.from_any("D4").to_dict()["col"], 3)

    def test_valid_json_payload(self):
        payload = json.dumps({
            "has_error": True,
            "summary": "wrong relation",
            "regions": [{"cells": ["B2"], "reason": "relation", "confidence": 0.8}],
            "correction_instruction": "move the red cube left",
        })
        evaluation = parse_semantic_evaluation(payload)
        self.assertTrue(evaluation.has_error)
        self.assertEqual(evaluation.regions[0].cells[0].to_label(), "B2")

    def test_invalid_json_safely_abstains(self):
        evaluation = safe_parse_semantic_evaluation("not json")
        self.assertTrue(evaluation.should_abstain)
        self.assertFalse(evaluation.has_error)

    def test_error_without_region_safely_abstains(self):
        evaluation = safe_parse_semantic_evaluation({"has_error": True, "regions": []})
        self.assertTrue(evaluation.should_abstain)

    def test_overbroad_selection_safely_abstains(self):
        regions = [{"cells": [f"R{row}C{col}"]} for row in range(4) for col in range(4)]
        evaluation = safe_parse_semantic_evaluation({"has_error": True, "regions": regions}, max_selected_cells=4)
        self.assertTrue(evaluation.should_abstain)


if __name__ == "__main__":
    unittest.main()

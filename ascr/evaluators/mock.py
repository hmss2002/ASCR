from ascr.core.schemas import safe_parse_semantic_evaluation


class MockSemanticEvaluator:
    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if iteration == 0:
            payload = {
                "has_error": True,
                "summary": "Mock localized semantic inconsistency in cell B2.",
                "regions": [
                    {
                        "cells": ["B2"],
                        "reason": "Mock region chosen to exercise Stage 1 reopening.",
                        "confidence": 0.9,
                        "error_type": "mock_semantic",
                    }
                ],
                "correction_instruction": "Repair the object relation in B2 while preserving all other grid cells.",
            }
        else:
            payload = {"has_error": False, "summary": "Mock evaluator reports no remaining semantic issue.", "regions": []}
        return safe_parse_semantic_evaluation(payload)

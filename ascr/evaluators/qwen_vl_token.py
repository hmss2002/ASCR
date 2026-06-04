from ascr.evaluators.qwen_vl import QwenVLEvaluator


class QwenVLTokenEvaluator(QwenVLEvaluator):
    """Direct token-level variant of :class:`QwenVLEvaluator`.

    Instead of asking the model to choose among the 16 cells of a 4x4 coarse
    grid (``A1``..``D4``), this evaluator asks it to directly point at which of
    the ``select_grid_size`` x ``select_grid_size`` discrete image tokens are
    wrong, using a numeric ``R{row}C{col}`` coordinate scheme (0-indexed). It
    reuses all of the parent's model-loading, generation, and parsing logic; the
    parent's grid_size is set to ``select_grid_size`` so parsed cells are
    validated at token resolution. The base :class:`QwenVLEvaluator` is left
    untouched.
    """

    def __init__(self, select_grid_size=32, **kwargs):
        self.select_grid_size = int(select_grid_size)
        kwargs["grid_size"] = self.select_grid_size
        super().__init__(**kwargs)

    def _coordinate_help(self):
        last = self.select_grid_size - 1
        return (
            f"The image is divided into a {self.select_grid_size}x{self.select_grid_size} grid of equally sized "
            f"cells. Reference grid lines and numeric ticks are drawn on the image as evaluation aids; do not "
            f"treat them as part of the scene. Identify each cell by the coordinate R{{row}}C{{col}} where row "
            f"and col are integers from 0 to {last}. Row 0 is the top, row {last} is the bottom; col 0 is the "
            f"left, col {last} is the right (for example R0C0 is the top-left cell and R{last}C{last} is the "
            f"bottom-right cell)."
        )

    def _build_question(self, original_prompt):
        parts = [
            "You are the strict semantic evaluator for ASCR Stage 1 direct-token reopening.",
            self._coordinate_help(),
            f"Original text-to-image prompt: {original_prompt}",
            "Decide whether the image materially violates the prompt. Check objects, counts, colors, attributes, text, and spatial relations.",
            "Return exactly one valid JSON object. Start with { and end with }.",
            "Do not include markdown, code fences, or bullet points.",
            'Use this schema: {"has_error": boolean, "summary": string, "regions": array, "correction_instruction": string}.',
            "If there is no material semantic error, set has_error to false and regions to an empty array.",
            f"If there is an error, select at most {self.max_selected_cells} R{{row}}C{{col}} cells (across all regions) that tightly cover the wrong, missing, or extra content; pick the smallest set of cells that still covers it.",
            "Each region must include cells (a list of R{row}C{col} strings), reason, confidence, error_type, and action set to reopen.",
        ]
        if self.enable_thinking:
            parts.insert(5, "The assistant response starts inside a <think> block; keep that block under 80 words, then close it with </think>.")
            parts.insert(6, "After </think>, write FINAL_JSON: followed immediately by exactly one valid JSON object.")
            parts.insert(7, "Do not write numbered or bulleted analysis.")
            parts.insert(8, "Stop immediately after the JSON object closing }.")
        else:
            parts.insert(0, "/no_think")
            parts.insert(5, "Do not include prose or analysis.")
        return " ".join(parts)

    def _build_json_repair_question(self, original_prompt, previous_text):
        previous_text = " ".join(str(previous_text).split())[:1800]
        return " ".join([
            "/no_think",
            "Return one compact valid JSON object for this semantic evaluation.",
            "Your entire response must be only the JSON object and must end with }.",
            f"Original text-to-image prompt: {original_prompt}",
            self._coordinate_help(),
            "Do not add prose, markdown, code fences, or analysis.",
            "Use this schema: {\"has_error\": boolean, \"summary\": string, \"regions\": array, \"correction_instruction\": string}.",
            "Keep summary under 25 words and each reason under 10 words.",
            "If the image satisfies the prompt, return has_error false and regions [].",
            f"If there is an error, select at most {self.max_selected_cells} R{{row}}C{{col}} cells across all regions; each region needs cells, reason, confidence, error_type, and action set to reopen.",
            "If the previous text is incomplete, finish the judgment instead of copying it.",
            "Previous incomplete evaluation text:",
            previous_text,
        ])

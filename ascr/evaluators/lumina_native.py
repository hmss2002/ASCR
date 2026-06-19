"""Lumina-native semantic evaluator scaffold.

This backend is intentionally conservative.  The current Lumina-DiMOO wrapper
already supports text-to-image generation, decoding, and token reopening, but it
does not yet expose a verified image-conditioned text/MMU answer method.  When
that method is absent, this evaluator abstains instead of guessing cells.
"""

from pathlib import Path

from ascr.core.schemas import SemanticEvaluation, safe_parse_semantic_evaluation
from ascr.distill.teacher import extract_json_object


ANSWER_IMAGE_METHODS = (
    "answer_image",
    "evaluate_image",
    "answer_grid_image",
)

ANSWER_TOKEN_METHODS = (
    "answer_vq_tokens",
    "evaluate_vq_tokens",
)


def native_eval_prompt(original_prompt, grid_size=4, max_selected_cells=6):
    cells = []
    for row in range(int(grid_size)):
        for col in range(int(grid_size)):
            cells.append(f"{chr(ord('A') + row)}{col + 1}")
    return (
        "You are the ASCR semantic evaluator for a generated image.\n"
        "Compare the current image against the original prompt.\n"
        "Return exactly one compact JSON object. No markdown. No analysis.\n"
        "Schema:\n"
        "{"
        "\"has_error\": boolean, "
        "\"summary\": string, "
        "\"regions\": [{\"cells\": [{\"label\": string}], \"reason\": string, "
        "\"confidence\": number, \"error_type\": string, \"action\": \"reopen\"}], "
        "\"correction_instruction\": string"
        "}\n"
        f"Allowed grid cells: {', '.join(cells)}.\n"
        f"Use at most {int(max_selected_cells)} selected cells.\n"
        "If the image already matches the prompt, set has_error=false and regions=[].\n"
        f"Original prompt: {original_prompt}"
    )


def supported_native_answer_methods(engine):
    methods = []
    for name in ANSWER_IMAGE_METHODS + ANSWER_TOKEN_METHODS:
        if callable(getattr(engine, name, None)):
            methods.append(name)
    return methods


def attach_lumina_native_engine_if_available(generator, evaluator):
    """Share a Lumina generator engine with the native evaluator when possible."""
    if not isinstance(evaluator, LuminaNativeEvaluator):
        return False
    if evaluator.engine is not None:
        return False
    engine_method = getattr(generator, "engine", None)
    if not callable(engine_method):
        return False
    try:
        evaluator.engine = engine_method()
    except Exception:
        evaluator.engine = None
        return False
    return evaluator.engine is not None


def _call_with_optional_max_tokens(method, *args, max_new_tokens=None):
    try:
        return method(*args, max_new_tokens=max_new_tokens)
    except TypeError:
        return method(*args)


def call_native_answer(engine, question, image_path=None, vq_ids=None, max_new_tokens=384):
    for name in ANSWER_IMAGE_METHODS:
        method = getattr(engine, name, None)
        if callable(method) and image_path:
            return str(_call_with_optional_max_tokens(method, question, str(image_path), max_new_tokens=max_new_tokens)), name
    for name in ANSWER_TOKEN_METHODS:
        method = getattr(engine, name, None)
        if callable(method) and vq_ids is not None:
            return str(_call_with_optional_max_tokens(method, question, list(vq_ids), max_new_tokens=max_new_tokens)), name
    raise NotImplementedError(
        "LuminaNativeEngine does not expose an image-conditioned text answer method. "
        "Expected one of: " + ", ".join(ANSWER_IMAGE_METHODS + ANSWER_TOKEN_METHODS)
    )


class LuminaNativeEvaluator:
    def __init__(
        self,
        checkpoint_path=None,
        repo_path=None,
        device="cuda",
        grid_size=4,
        image_size=1024,
        max_new_tokens=384,
        max_selected_cells=6,
        answer_steps=64,
        answer_block_length=128,
        answer_temperature=0.0,
        answer_cfg_scale=0.0,
        unsupported_policy="abstain",
        engine=None,
    ):
        self.grid_size = int(grid_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_selected_cells = int(max_selected_cells)
        self.answer_steps = int(answer_steps)
        self.answer_block_length = int(answer_block_length)
        self.answer_temperature = float(answer_temperature)
        self.answer_cfg_scale = float(answer_cfg_scale)
        self.unsupported_policy = str(unsupported_policy)
        self.engine = engine
        self.checkpoint_path = checkpoint_path
        self.repo_path = repo_path
        self.device = device
        self.image_size = int(image_size)

    def _engine(self):
        if self.engine is None:
            from ascr.generators.lumina_native import LuminaNativeEngine

            self.engine = LuminaNativeEngine(
                checkpoint_path=self.checkpoint_path or "models/lumina-dimoo",
                repo_path=self.repo_path,
                device=self.device,
                image_size=self.image_size,
                answer_steps=self.answer_steps,
                answer_block_length=self.answer_block_length,
                answer_temperature=self.answer_temperature,
                answer_cfg_scale=self.answer_cfg_scale,
            )
        return self.engine

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        prompt = current_prompt or original_prompt
        question = native_eval_prompt(prompt, grid_size=self.grid_size, max_selected_cells=self.max_selected_cells)
        try:
            raw_text, method_name = call_native_answer(
                self._engine(),
                question,
                image_path=Path(grid_image_path) if grid_image_path else None,
                max_new_tokens=self.max_new_tokens,
            )
        except NotImplementedError as exc:
            if self.unsupported_policy == "raise":
                raise
            return SemanticEvaluation.abstain(
                str(exc),
                raw={
                    "backend": "lumina_native_evaluator",
                    "iteration": int(iteration),
                    "supported_methods": supported_native_answer_methods(self._engine()),
                },
            )
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"Lumina native evaluator call failed: {exc}",
                raw={"backend": "lumina_native_evaluator", "iteration": int(iteration)},
            )

        try:
            payload = extract_json_object(raw_text)
            parsed = safe_parse_semantic_evaluation(
                payload,
                grid_size=self.grid_size,
                max_selected_cells=self.max_selected_cells,
            )
            parsed.raw = {
                "backend": "lumina_native_evaluator",
                "method": method_name,
                "raw_text": raw_text,
                "parsed": payload,
                "answer_steps": self.answer_steps,
                "answer_block_length": self.answer_block_length,
                "answer_temperature": self.answer_temperature,
                "answer_cfg_scale": self.answer_cfg_scale,
            }
            return parsed
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"Lumina native evaluator returned malformed JSON: {exc}",
                raw={"backend": "lumina_native_evaluator", "raw_text": raw_text},
            )

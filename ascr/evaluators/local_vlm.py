from ascr.evaluators.base import SemanticEvaluator


class LocalVLMEvaluator(SemanticEvaluator):
    def __init__(self, model_path=None, device="cuda", strict_json=True):
        self.model_path = model_path
        self.device = device
        self.strict_json = strict_json

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        raise NotImplementedError(
            "Local VLM evaluation is reserved behind SemanticEvaluator. Select evaluator=mock for dry runs or configure a concrete local VLM backend."
        )

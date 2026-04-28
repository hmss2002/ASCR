from ascr.evaluators.mock import MockSemanticEvaluator
from ascr.evaluators.local_vlm import LocalVLMEvaluator


def build_evaluator(name, config):
    name = (name or "mock").lower()
    config = config or {}
    if name == "mock":
        return MockSemanticEvaluator()
    if name in {"local_vlm", "local-vlm"}:
        evaluator_config = config.get("evaluator", config)
        return LocalVLMEvaluator(
            model_path=evaluator_config.get("model_path"),
            device=evaluator_config.get("device", "cuda"),
            strict_json=bool(evaluator_config.get("strict_json", True)),
        )
    raise ValueError(f"Unknown evaluator: {name}")

from ascr.evaluators.mock import MockSemanticEvaluator
from ascr.evaluators.local_vlm import LocalVLMEvaluator
from ascr.evaluators.showo_mmu import ShowOMMUEvaluator


def build_evaluator(name, config):
    name = (name or "mock").lower()
    config = config or {}
    if name == "mock":
        return MockSemanticEvaluator()
    if name in {"showo_mmu", "showo-mmu", "showo_vlm", "showo-vlm"}:
        evaluator_config = config.get("evaluator", config)
        generator_config = config.get("generator", {})
        return ShowOMMUEvaluator(
            repo_path=evaluator_config.get("repo_path", generator_config.get("repo_path", "external/Show-o")),
            checkpoint_path=evaluator_config.get("checkpoint_path", generator_config.get("checkpoint_path", "models/show-o-512x512")),
            vq_model_path=evaluator_config.get("vq_model_path", generator_config.get("vq_model_path", "models/magvitv2")),
            llm_model_path=evaluator_config.get("llm_model_path", generator_config.get("llm_model_path", "models/phi-1_5")),
            showo_config_path=evaluator_config.get("showo_config_path", generator_config.get("showo_config_path", "configs/showo_local_512x512.yaml")),
            device=evaluator_config.get("device", generator_config.get("device", "cuda")),
            grid_size=int(config.get("coarse_grid_size", evaluator_config.get("grid_size", 4))),
            image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
            max_new_tokens=int(evaluator_config.get("max_new_tokens", 192)),
        )
    if name in {"local_vlm", "local-vlm"}:
        evaluator_config = config.get("evaluator", config)
        backend = evaluator_config.get("backend", "heuristic")
        if backend in {"showo_mmu", "showo-mmu", "showo_vlm", "showo-vlm"}:
            return build_evaluator("showo_mmu", config)
        return LocalVLMEvaluator(
            model_path=evaluator_config.get("model_path"),
            device=evaluator_config.get("device", "cuda"),
            strict_json=bool(evaluator_config.get("strict_json", True)),
            backend=backend,
            grid_size=int(config.get("coarse_grid_size", evaluator_config.get("grid_size", 4))),
            image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
            pass_threshold=float(evaluator_config.get("pass_threshold", 0.62)),
        )
    raise ValueError(f"Unknown evaluator: {name}")

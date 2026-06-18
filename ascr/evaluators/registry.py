import os

from ascr.evaluators.mock import MockSemanticEvaluator


SHOWO_BACKENDS = {"showo", "showo_mmu", "showo-mmu"}
QWEN_BACKENDS = {"qwen", "qwen_vl", "qwen-vl"}
QWEN_TOKEN_BACKENDS = {"qwen_vl_token", "qwen-vl-token", "qwen_token", "qwen-token"}
MMADA_SELF_BACKENDS = {"mmada_self", "mmada-self", "mmada", "mmada_mmu", "mmada-mmu"}
MMADA_SELF_COARSE_BACKENDS = {"mmada_self_coarse", "mmada-self-coarse", "mmada_coarse", "mmada-coarse"}
STUDENT_LOCALIZER_BACKENDS = {"student_localizer", "student-localizer", "grid_localizer_v0", "grid-localizer-v0"}
LUMINA_NATIVE_BACKENDS = {
    "lumina_native_evaluator",
    "lumina-native-evaluator",
    "lumina_mmu",
    "lumina-mmu",
}


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_mmada_self(config):
    from ascr.evaluators.mmada_self import MMaDASelfEvaluator
    evaluator_config = config.get("evaluator", config)
    generator_config = config.get("generator", {})
    grid_size = int(evaluator_config.get("grid_size", config.get("select_grid_size", config.get("token_grid_size", 32))))
    return MMaDASelfEvaluator(
        repo_path=evaluator_config.get("repo_path", generator_config.get("repo_path", "external/MMaDA")),
        checkpoint_path=evaluator_config.get("checkpoint_path", generator_config.get("checkpoint_path", "models/mmada-8b-mixcot")),
        vq_model_path=evaluator_config.get("vq_model_path", generator_config.get("vq_model_path", "models/magvitv2")),
        device=evaluator_config.get("device", generator_config.get("device", "cuda")),
        grid_size=grid_size,
        image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
        max_new_tokens=int(evaluator_config.get("max_new_tokens", 256)),
        max_selected_cells=int(evaluator_config.get("max_selected_cells", config.get("selector", {}).get("max_selected_cells", 64))),
        confidence_fallback=bool(evaluator_config.get("confidence_fallback", True)),
        confidence_fallback_cells=evaluator_config.get("confidence_fallback_cells", None),
    )


def _build_mmada_self_coarse(config):
    from ascr.evaluators.mmada_self_coarse import MMaDASelfCoarseEvaluator
    evaluator_config = config.get("evaluator", config)
    generator_config = config.get("generator", {})
    grid_size = int(config.get("coarse_grid_size", evaluator_config.get("grid_size", 4)))
    token_grid_size = int(config.get("token_grid_size", evaluator_config.get("token_grid_size", 32)))
    return MMaDASelfCoarseEvaluator(
        repo_path=evaluator_config.get("repo_path", generator_config.get("repo_path", "external/MMaDA")),
        checkpoint_path=evaluator_config.get("checkpoint_path", generator_config.get("checkpoint_path", "models/mmada-8b-mixcot")),
        vq_model_path=evaluator_config.get("vq_model_path", generator_config.get("vq_model_path", "models/magvitv2")),
        device=evaluator_config.get("device", generator_config.get("device", "cuda")),
        grid_size=grid_size,
        token_grid_size=token_grid_size,
        image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
        max_new_tokens=int(evaluator_config.get("max_new_tokens", 48)),
        max_selected_cells=int(evaluator_config.get("max_selected_cells", config.get("selector", {}).get("max_selected_cells", 6))),
        confidence_fallback=bool(evaluator_config.get("confidence_fallback", True)),
        confidence_fallback_cells=evaluator_config.get("confidence_fallback_cells", None),
    )


def _build_showo_mmu(config):
    from ascr.evaluators.showo_mmu import ShowOMMUEvaluator

    evaluator_config = config.get("evaluator", config)
    generator_config = config.get("generator", {})
    return ShowOMMUEvaluator(
        repo_path=evaluator_config.get("repo_path", generator_config.get("repo_path", "external/Show-o")),
        checkpoint_path=evaluator_config.get("checkpoint_path", generator_config.get("checkpoint_path", "models/show-o-512x512")),
        vq_model_path=evaluator_config.get("vq_model_path", generator_config.get("vq_model_path", "models/magvitv2")),
        llm_model_path=evaluator_config.get("llm_model_path", generator_config.get("llm_model_path", "models/phi-1_5")),
        showo_config_path=evaluator_config.get("showo_config_path", generator_config.get("showo_config_path", "configs/stage1/showo/showo_local_512x512.yaml")),
        device=evaluator_config.get("device", generator_config.get("device", "cuda")),
        grid_size=int(config.get("coarse_grid_size", evaluator_config.get("grid_size", 4))),
        image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
        max_new_tokens=int(evaluator_config.get("max_new_tokens", 192)),
    )


def _build_qwen_vl(config):
    from ascr.evaluators.qwen_vl import QwenVLEvaluator

    evaluator_config = config.get("evaluator", config)
    return QwenVLEvaluator(
        model_path=os.environ.get("QWEN_MODEL_PATH", evaluator_config.get("model_path", "models/qwen3.5-9b")),
        device=evaluator_config.get("device", "cuda"),
        device_map=evaluator_config.get("device_map", "auto"),
        torch_dtype=evaluator_config.get("torch_dtype", "bfloat16"),
        trust_remote_code=_as_bool(evaluator_config.get("trust_remote_code", True), True),
        local_files_only=_as_bool(os.environ.get("QWEN_LOCAL_FILES_ONLY", evaluator_config.get("local_files_only", False)), False),
        strict_json=_as_bool(evaluator_config.get("strict_json", True), True),
        grid_size=int(config.get("coarse_grid_size", evaluator_config.get("grid_size", 4))),
        image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
        max_new_tokens=int(evaluator_config.get("max_new_tokens", 768)),
        repair_max_new_tokens=evaluator_config.get("repair_max_new_tokens"),
        max_selected_cells=int(evaluator_config.get("max_selected_cells", config.get("selector", {}).get("max_selected_cells", 6))),
        temperature=float(evaluator_config.get("temperature", 0.0)),
        top_p=float(evaluator_config.get("top_p", 1.0)),
        attn_implementation=evaluator_config.get("attn_implementation"),
        processor_use_fast=_as_bool(evaluator_config.get("processor_use_fast", False), False),
        enable_thinking=_as_bool(evaluator_config.get("enable_thinking", True), True),
        max_memory=os.environ.get("QWEN_MAX_MEMORY", evaluator_config.get("max_memory")),
    )


def _build_qwen_vl_token(config):
    from ascr.evaluators.qwen_vl_token import QwenVLTokenEvaluator

    evaluator_config = config.get("evaluator", config)
    select_grid_size = int(evaluator_config.get("select_grid_size", config.get("select_grid_size", config.get("token_grid_size", 32))))
    return QwenVLTokenEvaluator(
        select_grid_size=select_grid_size,
        model_path=os.environ.get("QWEN_MODEL_PATH", evaluator_config.get("model_path", "models/qwen3.5-9b")),
        device=evaluator_config.get("device", "cuda"),
        device_map=evaluator_config.get("device_map", "auto"),
        torch_dtype=evaluator_config.get("torch_dtype", "bfloat16"),
        trust_remote_code=_as_bool(evaluator_config.get("trust_remote_code", True), True),
        local_files_only=_as_bool(os.environ.get("QWEN_LOCAL_FILES_ONLY", evaluator_config.get("local_files_only", False)), False),
        strict_json=_as_bool(evaluator_config.get("strict_json", True), True),
        image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
        max_new_tokens=int(evaluator_config.get("max_new_tokens", 768)),
        repair_max_new_tokens=evaluator_config.get("repair_max_new_tokens"),
        max_selected_cells=int(evaluator_config.get("max_selected_cells", config.get("selector", {}).get("max_selected_cells", 64))),
        temperature=float(evaluator_config.get("temperature", 0.0)),
        top_p=float(evaluator_config.get("top_p", 1.0)),
        attn_implementation=evaluator_config.get("attn_implementation"),
        processor_use_fast=_as_bool(evaluator_config.get("processor_use_fast", False), False),
        enable_thinking=_as_bool(evaluator_config.get("enable_thinking", True), True),
        max_memory=os.environ.get("QWEN_MAX_MEMORY", evaluator_config.get("max_memory")),
    )


def build_evaluator(name, config):
    name = (name or "mock").lower()
    config = config or {}
    if name == "mock":
        return MockSemanticEvaluator()
    if name in SHOWO_BACKENDS:
        return _build_showo_mmu(config)
    if name in MMADA_SELF_COARSE_BACKENDS:
        return _build_mmada_self_coarse(config)
    if name in MMADA_SELF_BACKENDS:
        return _build_mmada_self(config)
    if name in QWEN_TOKEN_BACKENDS:
        return _build_qwen_vl_token(config)
    if name in QWEN_BACKENDS:
        return _build_qwen_vl(config)
    if name in STUDENT_LOCALIZER_BACKENDS:
        from ascr.evaluators.student_localizer import StudentLocalizerEvaluator

        evaluator_config = config.get("evaluator", config)
        return StudentLocalizerEvaluator(
            model_path=os.environ.get("STUDENT_MODEL", evaluator_config.get("model_path")),
            threshold=evaluator_config.get("threshold"),
            max_selected_cells=evaluator_config.get("max_selected_cells", config.get("selector", {}).get("max_selected_cells")),
            domain=os.environ.get("STUDENT_DOMAIN", evaluator_config.get("domain")),
        )
    if name in LUMINA_NATIVE_BACKENDS:
        from ascr.evaluators.lumina_native import LuminaNativeEvaluator

        evaluator_config = config.get("evaluator", config)
        generator_config = config.get("generator", {})
        return LuminaNativeEvaluator(
            checkpoint_path=evaluator_config.get("checkpoint_path", generator_config.get("checkpoint_path", "models/lumina-dimoo")),
            repo_path=evaluator_config.get("repo_path", generator_config.get("repo_path")),
            device=evaluator_config.get("device", generator_config.get("device", "cuda")),
            grid_size=int(config.get("coarse_grid_size", evaluator_config.get("grid_size", 4))),
            image_size=int(config.get("image_size", evaluator_config.get("image_size", generator_config.get("image_size", 1024)))),
            max_new_tokens=int(evaluator_config.get("max_new_tokens", 384)),
            max_selected_cells=int(evaluator_config.get("max_selected_cells", config.get("selector", {}).get("max_selected_cells", 6))),
            unsupported_policy=evaluator_config.get("unsupported_policy", "abstain"),
        )
    if name in {"local_vlm", "local-vlm"}:
        evaluator_config = config.get("evaluator", config)
        backend = str(evaluator_config.get("backend", "heuristic")).lower()
        if backend in SHOWO_BACKENDS:
            return _build_showo_mmu(config)
        if backend in QWEN_TOKEN_BACKENDS:
            return _build_qwen_vl_token(config)
        if backend in QWEN_BACKENDS:
            return _build_qwen_vl(config)
        from ascr.evaluators.local_vlm import LocalVLMEvaluator

        return LocalVLMEvaluator(
            model_path=evaluator_config.get("model_path"),
            device=evaluator_config.get("device", "cuda"),
            strict_json=_as_bool(evaluator_config.get("strict_json", True), True),
            backend=backend,
            grid_size=int(config.get("coarse_grid_size", evaluator_config.get("grid_size", 4))),
            image_size=int(config.get("image_size", evaluator_config.get("image_size", 512))),
            pass_threshold=float(evaluator_config.get("pass_threshold", 0.62)),
        )
    raise ValueError(f"Unknown evaluator: {name}")

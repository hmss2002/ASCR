import os

from ascr.evaluators.mock import MockSemanticEvaluator
from ascr.evaluators.local_vlm import LocalVLMEvaluator
from ascr.evaluators.qwen_vl import QwenVLEvaluator
from ascr.evaluators.qwen_vl_token import QwenVLTokenEvaluator
from ascr.evaluators.showo_mmu import ShowOMMUEvaluator


SHOWO_BACKENDS = {"showo_mmu", "showo-mmu", "showo_vlm", "showo-vlm"}
QWEN_BACKENDS = {"qwen", "qwen_vl", "qwen-vl", "qwen3_6", "qwen3.6", "qwen36", "qwen3_6_vl", "qwen3.6-vl"}
QWEN_TOKEN_BACKENDS = {"qwen_vl_token", "qwen-vl-token", "qwen_token", "qwen-token", "qwen3_6_token", "qwen36_token"}
MMADA_SELF_BACKENDS = {"mmada_self", "mmada-self", "mmada", "mmada_mmu", "mmada-mmu"}


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


def _build_showo_mmu(config):
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


def _build_qwen_vl(config):
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
    if name in MMADA_SELF_BACKENDS:
        return _build_mmada_self(config)
    if name in QWEN_TOKEN_BACKENDS:
        return _build_qwen_vl_token(config)
    if name in QWEN_BACKENDS:
        return _build_qwen_vl(config)
    if name in {"local_vlm", "local-vlm"}:
        evaluator_config = config.get("evaluator", config)
        backend = str(evaluator_config.get("backend", "heuristic")).lower()
        if backend in SHOWO_BACKENDS:
            return _build_showo_mmu(config)
        if backend in QWEN_TOKEN_BACKENDS:
            return _build_qwen_vl_token(config)
        if backend in QWEN_BACKENDS:
            return _build_qwen_vl(config)
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

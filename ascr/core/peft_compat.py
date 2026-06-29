"""Compatibility helpers for PEFT/Transformers version skew."""

from __future__ import annotations

import importlib
import sys
import types


def ensure_transformers_tensor_parallel_compat():
    """Install a tiny Transformers tensor_parallel stub when PEFT expects it.

    PEFT 0.19.x imports ``transformers.integrations.tensor_parallel`` on some
    adapter resume paths. Transformers 4.46.x does not ship that module. ASCR
    does not use tensor parallel plans here, so a no-op shim is enough to let
    PEFT take its early-return path without patching the server virtualenv.
    """

    module_name = "transformers.integrations.tensor_parallel"
    if module_name in sys.modules:
        return False
    try:
        importlib.import_module(module_name)
        return False
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    class _NoOpParallel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def gather_state_dict_for_save(model=None, state_dict=None, *args, **kwargs):
        if state_dict is not None:
            return state_dict
        if model is not None and hasattr(model, "state_dict"):
            return model.state_dict()
        return {}

    module = types.ModuleType(module_name)
    module.ALL_PARALLEL_STYLES = {}
    module.ColwiseParallel = _NoOpParallel
    module.RowwiseParallel = _NoOpParallel
    module.EmbeddingParallel = _NoOpParallel
    module.gather_state_dict_for_save = gather_state_dict_for_save
    module.__dict__["__all__"] = [
        "ALL_PARALLEL_STYLES",
        "ColwiseParallel",
        "RowwiseParallel",
        "EmbeddingParallel",
        "gather_state_dict_for_save",
    ]
    sys.modules[module_name] = module
    return True

import os
import sys
import unittest
from unittest.mock import patch

from ascr.core import peft_compat
from ascr.training.stage4_mmu_lora_ddp import (
    _ddp_constructor_options,
    _mark_ddp_ignored_frozen_parameters,
)


class _FakeDDP:
    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(model, ignored):
        model.ignored = list(ignored)

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        init_sync=False,
    ):
        self.module = module


class _FakeDevice:
    type = "cuda"


class _FakeParam:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad


class _FakeModel:
    def __init__(self):
        self.params = [
            ("base.weight", _FakeParam(False)),
            ("base.bias", _FakeParam(False)),
            ("adapter.lora_A.weight", _FakeParam(True)),
        ]
        self.ignored = []

    def named_parameters(self):
        return iter(self.params)


class Stage4DdpCompatTests(unittest.TestCase):
    def test_ddp_constructor_options_disable_constructor_sync_by_default(self):
        env = {"local_rank": 3}
        with patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ):
                if key.startswith("ASCR_DDP_"):
                    del os.environ[key]
            options, ignored = _ddp_constructor_options(_FakeDDP, env, _FakeDevice())
        self.assertEqual(ignored, [])
        self.assertEqual(options["device_ids"], [3])
        self.assertEqual(options["output_device"], 3)
        self.assertFalse(options["find_unused_parameters"])
        self.assertFalse(options["broadcast_buffers"])
        self.assertFalse(options["init_sync"])
        self.assertTrue(options["gradient_as_bucket_view"])

    def test_mark_ddp_ignored_frozen_parameters_keeps_lora_trainable(self):
        model = _FakeModel()
        report = _mark_ddp_ignored_frozen_parameters(_FakeDDP, model)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["ignored_parameter_count"], 2)
        self.assertEqual(model.ignored, ["base.weight", "base.bias"])

    def test_peft_tensor_parallel_compat_installs_noop_stub(self):
        module_name = "transformers.integrations.tensor_parallel"
        previous = sys.modules.pop(module_name, None)
        try:
            with patch.object(
                peft_compat.importlib,
                "import_module",
                side_effect=ModuleNotFoundError(name=module_name),
            ):
                installed = peft_compat.ensure_transformers_tensor_parallel_compat()
            self.assertTrue(installed)
            module = sys.modules[module_name]
            self.assertTrue(hasattr(module, "ColwiseParallel"))
            self.assertEqual(module.gather_state_dict_for_save(state_dict={"x": 1}), {"x": 1})
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous


if __name__ == "__main__":
    unittest.main()

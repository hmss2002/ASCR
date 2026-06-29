import os
import sys
import unittest
from unittest.mock import patch

from ascr.core import peft_compat
from ascr.training.stage4_mmu_lora_ddp import (
    _assert_rank_consistent_lora,
    _ddp_constructor_options,
    _mark_ddp_ignored_frozen_parameters,
)
from ascr.training.train_lumina_lora_smoke import _call_model_loss


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


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        value = self.values[index]
        if isinstance(value, list):
            return _FakeTensor(value)
        return value

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device=None, dtype=None):
        return self

    def tolist(self):
        return list(self.values)


class _FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return _FakeTensor(values)

    @staticmethod
    def zeros_like(tensor):
        return _FakeTensor([0 for _ in tensor.values])


class _FakeDist:
    def __init__(self, remote_values, backend="gloo"):
        self.remote_values = [list(values) for values in remote_values]
        self.backend = backend
        self.barrier_calls = []
        self.object_gather_called = False

    def get_backend(self):
        return self.backend

    def barrier(self, **kwargs):
        self.barrier_calls.append(kwargs)

    def all_gather(self, gathered_tensors, value_tensor):
        for target, values in zip(gathered_tensors, self.remote_values):
            target.values = list(values)

    def all_gather_object(self, reports, report):
        self.object_gather_called = True
        raise AssertionError("rank consistency should not use all_gather_object")


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


class _FakeLossOutput:
    def __init__(self, loss):
        self.loss = loss


class _FakeLuminaForwardModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, labels):
        self.calls.append((input_ids, labels))
        first_row = input_ids[0] if isinstance(input_ids, list) else input_ids
        if isinstance(first_row, _FakeTensor):
            raise TypeError("unsupported operand type(s) for +: 'Tensor' and 'list'")
        if isinstance(first_row, list):
            return _FakeLossOutput(1.25)
        raise TypeError(f"unexpected input type: {type(first_row)!r}")


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
        with patch.dict(
            os.environ,
            {"ASCR_DDP_DEBUG": "0", "ASCR_DDP_IGNORE_FROZEN_METHOD": "attribute"},
            clear=False,
        ):
            report = _mark_ddp_ignored_frozen_parameters(_FakeDDP, model)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["method"], "attribute")
        self.assertEqual(report["ignored_parameter_count"], 2)
        self.assertEqual(model._ddp_params_and_buffers_to_ignore, {"base.weight", "base.bias"})
        self.assertTrue(model.params[0][1]._ddp_ignored)
        self.assertTrue(model.params[1][1]._ddp_ignored)
        self.assertFalse(hasattr(model.params[2][1], "_ddp_ignored"))

    def test_mark_ddp_ignored_frozen_parameters_can_use_setter_method(self):
        model = _FakeModel()
        with patch.dict(
            os.environ,
            {"ASCR_DDP_DEBUG": "0", "ASCR_DDP_IGNORE_FROZEN_METHOD": "setter"},
            clear=False,
        ):
            report = _mark_ddp_ignored_frozen_parameters(_FakeDDP, model)
        self.assertEqual(report["method"], "setter")
        self.assertEqual(model.ignored, ["base.weight", "base.bias"])

    def test_rank_consistency_uses_tensor_gather(self):
        report = {
            "trainable_tensor_count": 2,
            "trainable_parameter_count": 16,
            "lora_tensor_count": 2,
            "lora_trainable_tensor_count": 2,
            "trainable_names_sample": ["adapter.lora_A.weight"],
            "lora_names_sample": ["adapter.lora_A.weight"],
        }
        dist = _FakeDist(remote_values=[[2, 16, 2, 2], [2, 16, 2, 2]])
        with patch.dict(
            os.environ,
            {"ASCR_DDP_DEBUG": "0", "ASCR_DDP_PRE_COLLECTIVE_BARRIER": "0"},
            clear=False,
        ):
            reports = _assert_rank_consistent_lora(
                _FakeTorch,
                dist,
                {"rank": 0, "local_rank": 0, "world_size": 2},
                report,
                "unused-output",
                _FakeDevice(),
            )
        self.assertFalse(dist.object_gather_called)
        self.assertEqual(dist.barrier_calls, [])
        self.assertEqual(reports[0]["trainable_parameter_count"], 16)
        self.assertEqual(reports[1]["lora_trainable_tensor_count"], 2)

    def test_rank_consistency_can_insert_nccl_pre_collective_barrier(self):
        report = {
            "trainable_tensor_count": 2,
            "trainable_parameter_count": 16,
            "lora_tensor_count": 2,
            "lora_trainable_tensor_count": 2,
        }
        dist = _FakeDist(remote_values=[[2, 16, 2, 2], [2, 16, 2, 2]], backend="nccl")
        with patch.dict(
            os.environ,
            {"ASCR_DDP_DEBUG": "0", "ASCR_DDP_PRE_COLLECTIVE_BARRIER": "1"},
            clear=False,
        ):
            _assert_rank_consistent_lora(
                _FakeTorch,
                dist,
                {"rank": 1, "local_rank": 1, "world_size": 2},
                report,
                "unused-output",
                _FakeDevice(),
            )
        self.assertEqual(dist.barrier_calls, [{"device_ids": [1]}])

    def test_call_model_loss_prefers_python_token_rows_for_lumina_forward(self):
        model = _FakeLuminaForwardModel()
        loss = _call_model_loss(
            _FakeTorch,
            model,
            _FakeTensor([[101, 102, 103]]),
            _FakeTensor([[-100, 102, 103]]),
            device="cuda:0",
        )
        self.assertEqual(loss, 1.25)
        self.assertEqual(model.calls[0][0], [[101, 102, 103]])

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

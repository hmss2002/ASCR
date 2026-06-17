import importlib.util
import io
from pathlib import Path
from contextlib import redirect_stdout
import unittest
from unittest import mock


def load_api_probe():
    path = Path(__file__).resolve().parents[1] / "scripts" / "distill" / "api_probe.py"
    spec = importlib.util.spec_from_file_location("ascr_test_api_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ApiProbeTests(unittest.TestCase):
    def test_empty_content_can_be_non_blocking(self):
        module = load_api_probe()
        with mock.patch.object(module, "build_client", return_value=object()), mock.patch.object(module, "chat_completion_text", side_effect=ValueError("empty API response content")), mock.patch.object(module, "api_settings", return_value={"model": "bailian/qwen3.7-plus", "base_url": "https://api.ofox.ai/v1"}):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(["--allow-empty-content"]), 0)

    def test_empty_content_is_blocking_by_default(self):
        module = load_api_probe()
        with mock.patch.object(module, "build_client", return_value=object()), mock.patch.object(module, "chat_completion_text", side_effect=ValueError("empty API response content")), mock.patch.object(module, "api_settings", return_value={"model": "bailian/qwen3.7-plus", "base_url": "https://api.ofox.ai/v1"}):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(module.main([]), 2)


if __name__ == "__main__":
    unittest.main()

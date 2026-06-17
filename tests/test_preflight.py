import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ascr.cli.preflight import run_preflight, scan_secrets


class PreflightTests(unittest.TestCase):
    def test_local_preflight_allows_missing_model_paths_as_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "output_dir: outputs/test",
                        "generator:",
                        "  name: lumina",
                        "  checkpoint_path: models/missing-lumina",
                        "  repo_path: third_party/missing-lumina",
                        "evaluator:",
                        "  name: qwen_vl",
                        "  model_path: models/missing-qwen",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = run_preflight("local", config_path=config_path, project_root=root)
            self.assertTrue(report["ok"], json.dumps(report, indent=2))
            self.assertTrue(any(item["level"] == "warn" for item in report["records"]))

    def test_secret_scan_ignores_archived_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs" / "archive"
            docs.mkdir(parents=True)
            env_name = "OFOX_" + "API_KEY"
            fake_key = "sk" + "-of-" + "1234567890abcdef"
            (docs / "old.md").write_text(f"export {env_name}=\"{fake_key}\"\n", encoding="utf-8")
            script_dir = root / "scripts"
            script_dir.mkdir()
            safe_line = "export " + env_name + "=${" + env_name + ":?set key}\n"
            (script_dir / "safe.sh").write_text(safe_line, encoding="utf-8")
            self.assertEqual(scan_secrets(root), [])

    def test_registry_import_does_not_import_optional_vlm_modules(self):
        code = (
            "import json, sys; "
            "import ascr.evaluators.registry; "
            "print(json.dumps({"
            "'local_vlm': 'ascr.evaluators.local_vlm' in sys.modules, "
            "'qwen_vl': 'ascr.evaluators.qwen_vl' in sys.modules, "
            "'showo_mmu': 'ascr.evaluators.showo_mmu' in sys.modules"
            "}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = json.loads(completed.stdout)
        self.assertFalse(loaded["local_vlm"])
        self.assertFalse(loaded["qwen_vl"])
        self.assertFalse(loaded["showo_mmu"])


if __name__ == "__main__":
    unittest.main()

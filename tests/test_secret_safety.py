import json
import unittest
from pathlib import Path

from ascr.cli.preflight import scan_secrets


class SecretSafetyTests(unittest.TestCase):
    def test_tracked_text_files_do_not_contain_likely_committed_secrets(self):
        findings = scan_secrets(Path.cwd())
        self.assertEqual(findings, [], json.dumps(findings, indent=2))


if __name__ == "__main__":
    unittest.main()

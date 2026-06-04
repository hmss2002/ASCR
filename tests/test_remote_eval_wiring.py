import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ascr.evaluators.remote_eval import RemoteFileEvaluator, _evaluation_from_dict


class RemoteEvalReconstructTests(unittest.TestCase):
    def test_from_dict_error_with_regions(self):
        payload = {
            "has_error": True,
            "summary": "blue and red swapped",
            "regions": [{"cells": [{"row": 0, "col": 1}, {"row": 3, "col": 3}], "action": "reopen"}],
        }
        ev = _evaluation_from_dict(payload, 4)
        self.assertTrue(ev.has_error)
        cells = {(c.row, c.col) for r in ev.regions for c in r.cells}
        self.assertEqual(cells, {(0, 1), (3, 3)})

    def test_from_dict_abstain(self):
        ev = _evaluation_from_dict({"should_abstain": True, "parser_error": "boom"}, 4)
        self.assertTrue(ev.should_abstain)
        self.assertFalse(ev.has_error)

    def test_from_dict_no_error(self):
        ev = _evaluation_from_dict({"has_error": False, "summary": "looks good"}, 4)
        self.assertFalse(ev.has_error)
        self.assertEqual(ev.regions, [])

    def test_from_dict_error_without_regions_becomes_abstain(self):
        ev = _evaluation_from_dict({"has_error": True, "summary": "x", "regions": []}, 4)
        self.assertTrue(ev.should_abstain)


class RemoteEvalIPCTests(unittest.TestCase):
    def test_request_response_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ipc = Path(tmp) / "ipc"
            ev = RemoteFileEvaluator(ipc, grid_size=4, request_timeout=5.0, poll_interval=0.02)
            from PIL import Image
            img = Path(tmp) / "grid.png"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(img)

            def fake_server():
                req_dir = ipc / "requests"
                resp_dir = ipc / "responses"
                deadline = time.time() + 5
                while time.time() < deadline:
                    pending = [p for p in req_dir.glob("r*.json") if not p.name.endswith(".tmp")]
                    if pending:
                        req = pending[0]
                        payload = {"has_error": True, "summary": "s",
                                   "regions": [{"cells": [{"row": 1, "col": 2}], "action": "reopen"}]}
                        (resp_dir / req.name).write_text(json.dumps(payload))
                        return
                    time.sleep(0.02)

            t = threading.Thread(target=fake_server)
            t.start()
            result = ev.evaluate("a scene", str(img), 0)
            t.join()
            self.assertTrue(result.has_error)
            cells = {(c.row, c.col) for r in result.regions for c in r.cells}
            self.assertEqual(cells, {(1, 2)})

    def test_timeout_abstains(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = RemoteFileEvaluator(Path(tmp) / "ipc", grid_size=4, request_timeout=0.2, poll_interval=0.02)
            from PIL import Image
            img = Path(tmp) / "grid.png"
            Image.new("RGB", (64, 64), (0, 0, 0)).save(img)
            result = ev.evaluate("a scene", str(img), 0)
            self.assertTrue(result.should_abstain)


if __name__ == "__main__":
    unittest.main()

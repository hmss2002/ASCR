import base64
import json
from pathlib import Path
import tempfile
import unittest

from ascr.distill.teacher import build_tasks, extract_json_object, normalize_quality, run_task


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def create(self, **kwargs):
        text = json.dumps(kwargs["messages"])
        if "Image A is the baseline" in text:
            return _Response('{"baseline_score": 0.25, "final_score": 0.75, "winner": "B", "reason": "final follows the prompt better"}')
        return _Response('{"has_error": true, "summary": "wrong color", "regions": [{"cells": ["B2"], "reason": "wrong object color", "confidence": 0.9, "error_type": "attribute", "action": "reopen"}], "correction_instruction": "fix the object color"}')


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _Client:
    def __init__(self):
        self.chat = _Chat()


class TeacherDistillTests(unittest.TestCase):
    def test_extract_json_object_handles_markdown(self):
        payload = extract_json_object("```json\n{\"ok\": true}\n```")
        self.assertEqual(payload, {"ok": True})

    def test_normalize_quality_maps_b_to_final(self):
        payload = normalize_quality({"baseline_score": 0.2, "final_score": 0.8, "winner": "B"})
        self.assertEqual(payload["winner"], "final")
        self.assertEqual(payload["final_score"], 0.8)

    def test_build_tasks_from_stage1_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out = root / "outputs" / "run"
            (out / "records").mkdir(parents=True)
            (out / "baseline").mkdir()
            (out / "self").mkdir()
            (out / "baseline" / "p000.png").write_bytes(PNG_1X1)
            (out / "self" / "p000.png").write_bytes(PNG_1X1)
            record = {
                "idx": 0,
                "prompt": "A red cube",
                "baseline_image": str(out / "baseline" / "p000.png"),
                "final_image": str(out / "self" / "p000.png"),
            }
            (out / "records" / "p000.json").write_text(json.dumps(record), encoding="utf-8")
            run = out / "runs" / "p000" / "stage1-test"
            run.mkdir(parents=True)
            (run / "grid.png").write_bytes(PNG_1X1)
            trace = {
                "iteration": 0,
                "original_prompt": "A red cube",
                "current_prompt": "A red cube",
                "artifact_paths": {"grid_image": str(run / "grid.png")},
            }
            (run / "trace.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")

            tasks = build_tasks(out, root, limit=1)
            self.assertEqual([task["kind"] for task in tasks], ["quality", "localization"])
            self.assertEqual(tasks[0]["sample_id"], "p000")
            self.assertEqual(tasks[1]["sample_id"], "p000:i000")

    def test_run_task_with_fake_client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "img.png"
            image.write_bytes(PNG_1X1)
            client = _Client()
            quality = run_task(
                {"kind": "quality", "sample_id": "p000", "prompt": "A red cube", "baseline_image": str(image), "final_image": str(image)},
                client,
                "fake-model",
                4,
                6,
                1,
            )
            self.assertEqual(quality["quality"]["winner"], "final")
            localization = run_task(
                {"kind": "localization", "sample_id": "p000:i000", "prompt": "A red cube", "grid_image": str(image)},
                client,
                "fake-model",
                4,
                6,
                1,
            )
            self.assertTrue(localization["evaluation"]["has_error"])
            self.assertEqual(localization["evaluation"]["regions"][0]["cells"][0]["label"], "B2")


if __name__ == "__main__":
    unittest.main()


import base64
import json
from pathlib import Path
import tempfile
import unittest

from ascr.distill.audit import audit_distill_dir
from ascr.distill.export_dataset import export_dataset
from ascr.distill.teacher import (
    TeacherJsonParseError,
    build_tasks,
    error_payload,
    extract_json_object,
    extract_json_object_with_repair,
    localization_messages,
    normalize_quality,
    prune_resolved_errors,
    quality_messages,
    run_task,
)
from ascr.training.train_selector import train_cell_prior


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
        if "baseline_score" in text:
            return _Response('{"baseline_score": 0.25, "final_score": 0.75, "winner": "B", "reason": "final follows the prompt better"}')
        return _Response('{"has_error": true, "summary": "wrong color", "regions": [{"cells": ["B2"], "reason": "wrong object color", "confidence": 0.9, "error_type": "attribute", "action": "reopen"}], "correction_instruction": "fix the object color"}')


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _Client:
    def __init__(self):
        self.chat = _Chat()


class _RepairCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Response('{"has_error": false, "summary": "abstain", "regions": [], "correction_instruction": ""}')


class _RepairClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _RepairCompletions()})()


class _EmptyRepairCompletions:
    def create(self, **kwargs):
        return _Response("")


class _EmptyRepairClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _EmptyRepairCompletions()})()


class TeacherDistillTests(unittest.TestCase):
    def test_extract_json_object_handles_markdown(self):
        payload = extract_json_object("```json\n{\"ok\": true}\n```")
        self.assertEqual(payload, {"ok": True})

    def test_extract_json_object_uses_trailing_json(self):
        payload = extract_json_object("thinking text {\"bad\": true} more text {\"ok\": true}")
        self.assertEqual(payload, {"ok": True})

    def test_json_repair_falls_back_to_valid_object(self):
        payload = extract_json_object_with_repair(
            "The image appears correct.",
            _RepairClient(),
            "fake-model",
            '{"has_error": boolean, "summary": string, "regions": array, "correction_instruction": string}',
            repair_retries=1,
        )
        self.assertEqual(payload["has_error"], False)

    def test_json_repair_uses_local_abstention_when_text_only_repair_is_empty(self):
        payload = extract_json_object_with_repair(
            "The image appears correct.",
            _EmptyRepairClient(),
            "fake-model",
            '{"has_error": boolean, "summary": string, "regions": array, "correction_instruction": string}',
            repair_retries=1,
        )
        self.assertEqual(payload["has_error"], False)
        self.assertTrue(payload["should_abstain"])
        self.assertEqual(payload["repair_fallback"]["type"], "local-empty-json-repair")

    def test_error_payload_keeps_raw_preview_for_parse_failures(self):
        exc = TeacherJsonParseError("bad json", raw_text="natural language answer")
        payload = error_payload({"kind": "localization", "sample_id": "p001:i000"}, exc, "fake")
        self.assertEqual(payload["error_type"], "TeacherJsonParseError")
        self.assertEqual(payload["raw_preview"], "natural language answer")
        self.assertNotIn("raw_text", payload)

    def test_compact_prompt_constraints_are_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "img.png"
            image.write_bytes(PNG_1X1)
            text = quality_messages("A red cube", image, image)[1]["content"][0]["text"]
            self.assertIn("No analysis", text)
            self.assertIn("No markdown", text)
            self.assertIn("No thinking text", text)
            loc_text = localization_messages("A red cube", image)[1]["content"][0]["text"]
            self.assertIn("Return only one compact JSON object", loc_text)

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
            self.assertFalse(Path(tasks[0]["baseline_image"]).is_absolute())

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
            self.assertNotIn("raw_text", quality)
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

    def test_audit_export_and_cell_prior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            distill = root / "distill"
            distill.mkdir()
            quality = {
                "idx": 0,
                "sample_id": "p000",
                "prompt": "A red cube",
                "kind": "quality",
                "quality": {"baseline_score": 0.2, "final_score": 0.8, "winner": "final", "reason": "better"},
            }
            localization = {
                "idx": 0,
                "sample_id": "p000:i000",
                "prompt": "A red cube",
                "kind": "localization",
                "iteration": 0,
                "evaluation": {
                    "has_error": True,
                    "regions": [{"cells": [{"label": "B2", "row": 1, "col": 1}]}],
                },
            }
            (distill / "quality_labels.jsonl").write_text(json.dumps(quality) + "\n", encoding="utf-8")
            (distill / "localization_labels.jsonl").write_text(json.dumps(localization) + "\n", encoding="utf-8")
            old_error = {"kind": "localization", "sample_id": "p000:i000", "error": "old parse failure"}
            unresolved_error = {"kind": "localization", "sample_id": "p000:i001", "error": "still missing"}
            (distill / "errors.jsonl").write_text(json.dumps(old_error) + "\n" + json.dumps(unresolved_error) + "\n", encoding="utf-8")
            audit = audit_distill_dir(distill)
            self.assertEqual(audit["counts"]["quality"], 1)
            self.assertEqual(audit["counts"]["errors"], 2)
            self.assertEqual(audit["counts"]["unresolved_errors"], 1)
            self.assertEqual(audit["errors"]["unresolved_sample_ids"], ["p000:i001"])
            self.assertEqual(audit["localization"]["selected_cell_counts"]["1"], 1)
            manifest = export_dataset(distill, distill / "dataset.jsonl")
            self.assertEqual(manifest["row_count"], 1)
            train = train_cell_prior(distill / "dataset.jsonl", root / "baseline")
            self.assertEqual(train["metrics"]["hit_any_rate"], 1.0)
            self.assertTrue((root / "baseline" / "selector_prior.json").exists())

    def test_cell_prior_holdout_is_seeded_and_top_k_configurable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.jsonl"
            rows = []
            labels = ["A1", "B2", "C3", "D4", "B2"]
            for idx, label in enumerate(labels):
                rows.append({
                    "idx": idx,
                    "sample_id": f"p{idx:03d}",
                    "prompt": f"prompt {idx}",
                    "localizations": [{
                        "evaluation": {
                            "has_error": True,
                            "regions": [{"cells": [{"label": label}]}],
                        }
                    }],
                })
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = train_cell_prior(dataset, root / "holdout", eval_mode="holdout", train_ratio=0.6, seed=7, top_k=2)
            self.assertEqual(result["metrics"]["eval_mode"], "holdout")
            self.assertEqual(result["metrics"]["top_k"], 2)
            self.assertTrue((root / "holdout" / "split_manifest.json").exists())
            predictions = [json.loads(line) for line in (root / "holdout" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(predictions)
            self.assertTrue(all(len(row["predicted_cells"]) <= 2 for row in predictions))
            repeat = train_cell_prior(dataset, root / "holdout_repeat", eval_mode="holdout", train_ratio=0.6, seed=7, top_k=2)
            split = json.loads((root / "holdout" / "split_manifest.json").read_text(encoding="utf-8"))
            split_repeat = json.loads((root / "holdout_repeat" / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(split["eval_indices"], split_repeat["eval_indices"])
            self.assertEqual(result["metrics"]["eval_rows"], repeat["metrics"]["eval_rows"])

    def test_cell_prior_missing_dataset_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                train_cell_prior(root / "missing.jsonl", root / "should_not_exist")
            self.assertFalse((root / "should_not_exist").exists())

    def test_compute_training_scripts_do_not_reference_ofox(self):
        repo = Path(__file__).resolve().parents[1]
        script = (repo / "scripts" / "training" / "run_cell_prior.sh").read_text(encoding="utf-8")
        sbatch = (repo / "jobs" / "training" / "stage2_cell_prior_baseline.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("OFOX", script)
        self.assertNotIn("OFOX", sbatch)
        self.assertIn("--task cell-prior", script)

    def test_prune_resolved_errors_drops_resolved_rows_and_dedupes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            errors_path = Path(temp_dir) / "errors.jsonl"
            errors_path.write_text(
                "\n".join([
                    json.dumps({"sample_id": "p000:i000", "error": "old"}),
                    json.dumps({"sample_id": "p001:i000", "error": "older"}),
                    json.dumps({"sample_id": "p000:i000", "error": "new"}),
                    json.dumps({"sample_id": "p001:i000", "error": "newest"}),
                ]) + "\n",
                encoding="utf-8",
            )
            kept = prune_resolved_errors(errors_path, {"p000:i000"})
            self.assertEqual(kept, 1)
            rows = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows, [{"sample_id": "p001:i000", "error": "newest"}])


if __name__ == "__main__":
    unittest.main()

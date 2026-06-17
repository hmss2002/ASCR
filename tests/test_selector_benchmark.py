import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from ascr.benchmarks.selector_benchmark import run_selector_benchmark


class SelectorBenchmarkTests(unittest.TestCase):
    def test_labeled_and_unlabeled_domains(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selector = root / "selector_prior.json"
            selector.write_text(json.dumps({
                "schema_version": "ascr.cell_prior_selector.v1",
                "top_k": 2,
                "cell_counts": {"B2": 3, "C3": 2, "A1": 1},
            }), encoding="utf-8")
            dataset = root / "dataset.jsonl"
            rows = [
                {"sample_id": "p000", "prompt": "in domain 0", "localizations": [{"evaluation": {"regions": [{"cells": [{"label": "B2"}]}]}}]},
                {"sample_id": "p001", "prompt": "in domain 1", "localizations": [{"evaluation": {"regions": [{"cells": [{"label": "D4"}]}]}}]},
                {"sample_id": "p002", "prompt": "in domain 2", "localizations": []},
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            split = root / "split_manifest.json"
            split.write_text(json.dumps({"eval_sample_ids": ["p000", "p001"]}), encoding="utf-8")
            prompts = root / "drawbench.txt"
            prompts.write_text("A blue cube\nA red sphere\n", encoding="utf-8")
            output = root / "bench"

            report = run_selector_benchmark(Namespace(
                selector=selector,
                output_dir=output,
                top_k=None,
                in_domain_dataset=dataset,
                in_domain_split=split,
                out_domain_dataset=None,
                out_domain_split=None,
                out_domain_prompts=prompts,
                out_domain_limit=1,
            ))

            self.assertEqual(report["predicted_cells"], ["B2", "C3"])
            self.assertEqual(report["domains"]["in_domain"]["evaluated_rows"], 2)
            self.assertEqual(report["domains"]["in_domain"]["hit_any"], 1)
            self.assertEqual(report["domains"]["out_domain"]["label_status"], "unlabeled_prompts_only")
            self.assertTrue((output / "benchmark_report.json").exists())
            out_rows = [json.loads(line) for line in (output / "out_domain_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(out_rows), 1)
            self.assertIsNone(out_rows[0]["target_cells"])


if __name__ == "__main__":
    unittest.main()

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

from ascr.training.ddp import get_distributed_context


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def localization_cells(localization):
    cells = []
    evaluation = localization.get("evaluation", {})
    for region in evaluation.get("regions", []) or []:
        for cell in region.get("cells", []) or []:
            label = cell.get("label") if isinstance(cell, dict) else str(cell)
            if label:
                cells.append(str(label))
    return cells


def train_cell_prior(dataset_path, output_dir):
    rows = read_jsonl(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_counts = Counter()
    positive_rows = 0
    total_localizations = 0
    for row in rows:
        row_cells = []
        for localization in row.get("localizations", []) or []:
            total_localizations += 1
            row_cells.extend(localization_cells(localization))
        if row_cells:
            positive_rows += 1
        cell_counts.update(row_cells)
    total_cells = sum(cell_counts.values())
    prior = {
        "schema_version": "ascr.cell_prior_selector.v1",
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": str(dataset_path),
        "row_count": len(rows),
        "total_localizations": total_localizations,
        "positive_rows": positive_rows,
        "cell_counts": dict(sorted(cell_counts.items())),
        "cell_probabilities": {cell: count / total_cells for cell, count in sorted(cell_counts.items())} if total_cells else {},
    }
    (output_dir / "selector_prior.json").write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions = []
    for row in rows:
        target = sorted({cell for localization in row.get("localizations", []) or [] for cell in localization_cells(localization)})
        predicted = [cell for cell, _ in cell_counts.most_common(max(1, min(3, len(cell_counts))))]
        predictions.append({
            "idx": row.get("idx"),
            "sample_id": row.get("sample_id"),
            "prompt": row.get("prompt"),
            "target_cells": target,
            "predicted_cells": predicted,
            "hit_any": bool(set(target) & set(predicted)) if target else None,
        })
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            json.dump(prediction, handle, sort_keys=True)
            handle.write("\n")
    evaluated = [item for item in predictions if item["hit_any"] is not None]
    hits = sum(1 for item in evaluated if item["hit_any"])
    metrics = {
        "schema_version": "ascr.cell_prior_metrics.v1",
        "dataset": str(dataset_path),
        "row_count": len(rows),
        "evaluated_rows": len(evaluated),
        "hit_any": hits,
        "hit_any_rate": hits / len(evaluated) if evaluated else None,
        "top_cells": cell_counts.most_common(8),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"prior": prior, "metrics": metrics, "output_dir": str(output_dir)}


def build_parser():
    parser = argparse.ArgumentParser(description="ASCR selector training and baselines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=["reserved", "cell-prior"], default="reserved")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default="outputs/stage2_baselines/cell_prior")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "cell-prior":
        if not args.dataset:
            raise SystemExit("--dataset is required for --task cell-prior")
        result = train_cell_prior(args.dataset, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    payload = {"status": "reserved_for_stage2", "config": args.config, "distributed": get_distributed_context()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit("Stage 2 learned selector training is not implemented yet; use --task cell-prior for the lightweight baseline.")


if __name__ == "__main__":
    raise SystemExit(main())

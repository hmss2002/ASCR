import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random

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


def split_rows(rows, eval_mode="resubstitution", train_ratio=0.8, seed=0):
    if eval_mode == "resubstitution":
        indices = list(range(len(rows)))
        return indices, indices
    if eval_mode != "holdout":
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    if not rows:
        return [], []
    if len(rows) == 1:
        return [0], [0]
    rng = random.Random(int(seed))
    positive_indices = []
    negative_indices = []
    for index, row in enumerate(rows):
        target = [cell for localization in row.get("localizations", []) or [] for cell in localization_cells(localization)]
        if target:
            positive_indices.append(index)
        else:
            negative_indices.append(index)

    def split_group(indices):
        if not indices:
            return [], []
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            return shuffled, []
        train_count = int(len(shuffled) * float(train_ratio))
        train_count = max(1, min(len(shuffled) - 1, train_count))
        return shuffled[:train_count], shuffled[train_count:]

    train_pos, eval_pos = split_group(positive_indices)
    train_neg, eval_neg = split_group(negative_indices)
    train_indices = sorted(train_pos + train_neg)
    eval_indices = sorted(eval_pos + eval_neg)
    if not eval_indices:
        eval_indices = train_indices
    return train_indices, eval_indices


def build_cell_counts(rows):
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
    return cell_counts, positive_rows, total_localizations


def train_cell_prior(dataset_path, output_dir, eval_mode="resubstitution", train_ratio=0.8, seed=0, top_k=3):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    rows = read_jsonl(dataset_path)
    train_indices, eval_indices = split_rows(rows, eval_mode=eval_mode, train_ratio=train_ratio, seed=seed)
    train_rows = [rows[index] for index in train_indices]
    eval_rows = [rows[index] for index in eval_indices]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_counts, positive_rows, total_localizations = build_cell_counts(train_rows)
    total_cells = sum(cell_counts.values())
    top_k = max(1, int(top_k))
    predicted_template = [cell for cell, _ in cell_counts.most_common(max(1, min(top_k, len(cell_counts))))] if cell_counts else []
    prior = {
        "schema_version": "ascr.cell_prior_selector.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": str(dataset_path),
        "eval_mode": eval_mode,
        "row_count": len(train_rows),
        "source_row_count": len(rows),
        "top_k": top_k,
        "total_localizations": total_localizations,
        "positive_rows": positive_rows,
        "cell_counts": dict(sorted(cell_counts.items())),
        "cell_probabilities": {cell: count / total_cells for cell, count in sorted(cell_counts.items())} if total_cells else {},
    }
    (output_dir / "selector_prior.json").write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions = []
    for row in eval_rows:
        target = sorted({cell for localization in row.get("localizations", []) or [] for cell in localization_cells(localization)})
        predictions.append({
            "idx": row.get("idx"),
            "sample_id": row.get("sample_id"),
            "prompt": row.get("prompt"),
            "target_cells": target,
            "predicted_cells": predicted_template,
            "hit_any": bool(set(target) & set(predicted_template)) if target else None,
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
        "eval_mode": eval_mode,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "top_k": top_k,
        "evaluated_rows": len(evaluated),
        "hit_any": hits,
        "hit_any_rate": hits / len(evaluated) if evaluated else None,
        "top_cells": cell_counts.most_common(8),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if eval_mode == "holdout":
        split_manifest = {
            "schema_version": "ascr.cell_prior_split.v1",
            "dataset": str(dataset_path),
            "seed": int(seed),
            "train_ratio": float(train_ratio),
            "row_count": len(rows),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "train_sample_ids": [rows[index].get("sample_id") for index in train_indices],
            "eval_sample_ids": [rows[index].get("sample_id") for index in eval_indices],
        }
        (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"prior": prior, "metrics": metrics, "output_dir": str(output_dir)}


def build_parser():
    parser = argparse.ArgumentParser(description="ASCR selector training and baselines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=["reserved", "cell-prior"], default="reserved")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default="outputs/stage2_baselines/cell_prior")
    parser.add_argument("--eval-mode", choices=["resubstitution", "holdout"], default="resubstitution")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "cell-prior":
        if not args.dataset:
            raise SystemExit("--dataset is required for --task cell-prior")
        result = train_cell_prior(
            args.dataset,
            args.output_dir,
            eval_mode=args.eval_mode,
            train_ratio=args.train_ratio,
            seed=args.seed,
            top_k=args.top_k,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    payload = {"status": "reserved_for_stage2", "config": args.config, "distributed": get_distributed_context()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit("Stage 2 learned selector training is not implemented yet; use --task cell-prior for the lightweight baseline.")


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path

from ascr.training.train_selector import localization_cells


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def load_selector_cells(selector_path, top_k=None):
    payload = json.loads(Path(selector_path).read_text(encoding="utf-8"))
    counts = payload.get("cell_counts", {})
    if counts:
        ordered = [cell for cell, _ in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))]
    else:
        probabilities = payload.get("cell_probabilities", {})
        ordered = [cell for cell, _ in sorted(probabilities.items(), key=lambda item: (-float(item[1]), item[0]))]
    if not ordered:
        raise ValueError(f"Selector has no cell prior entries: {selector_path}")
    k = int(top_k if top_k is not None else payload.get("top_k", 3))
    return ordered[: max(1, min(k, len(ordered)))]


def target_cells(row):
    return sorted({cell for localization in row.get("localizations", []) or [] for cell in localization_cells(localization)})


def split_filter(rows, split_path=None):
    if not split_path:
        return rows
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    eval_ids = {str(item) for item in split.get("eval_sample_ids", []) if item is not None}
    if eval_ids:
        return [row for row in rows if str(row.get("sample_id")) in eval_ids]
    eval_indices = set(int(index) for index in split.get("eval_indices", []))
    return [row for offset, row in enumerate(rows) if offset in eval_indices]


def evaluate_labeled(rows, predicted_cells, domain):
    predictions = []
    for row in rows:
        target = target_cells(row)
        hit_any = bool(set(target) & set(predicted_cells)) if target else None
        predictions.append({
            "domain": domain,
            "idx": row.get("idx"),
            "sample_id": row.get("sample_id"),
            "prompt": row.get("prompt"),
            "target_cells": target,
            "predicted_cells": predicted_cells,
            "hit_any": hit_any,
        })
    evaluated = [row for row in predictions if row["hit_any"] is not None]
    hits = sum(1 for row in evaluated if row["hit_any"])
    metrics = {
        "domain": domain,
        "label_status": "labeled",
        "row_count": len(rows),
        "evaluated_rows": len(evaluated),
        "hit_any": hits,
        "hit_any_rate": hits / len(evaluated) if evaluated else None,
    }
    return metrics, predictions


def read_prompts(path, limit=None):
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            prompts.append(text)
        if limit is not None and len(prompts) >= int(limit):
            break
    return prompts


def predict_unlabeled_prompts(prompt_path, predicted_cells, domain, limit=None):
    prompts = read_prompts(prompt_path, limit=limit)
    rows = [
        {
            "domain": domain,
            "sample_id": f"{domain}:{index:04d}",
            "prompt": prompt,
            "predicted_cells": predicted_cells,
            "target_cells": None,
            "hit_any": None,
        }
        for index, prompt in enumerate(prompts)
    ]
    metrics = {
        "domain": domain,
        "label_status": "unlabeled_prompts_only",
        "prompt_file": str(prompt_path),
        "row_count": len(rows),
        "evaluated_rows": 0,
        "hit_any": None,
        "hit_any_rate": None,
    }
    return metrics, rows


def run_selector_benchmark(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_cells = load_selector_cells(args.selector, top_k=args.top_k)
    report = {
        "schema_version": "ascr.selector_benchmark.v1",
        "selector": str(args.selector),
        "predicted_cells": predicted_cells,
        "domains": {},
    }
    if args.in_domain_dataset:
        rows = split_filter(read_jsonl(args.in_domain_dataset), split_path=args.in_domain_split)
        metrics, predictions = evaluate_labeled(rows, predicted_cells, "in_domain")
        report["domains"]["in_domain"] = metrics
        write_jsonl(output_dir / "in_domain_predictions.jsonl", predictions)
    if args.out_domain_dataset:
        rows = split_filter(read_jsonl(args.out_domain_dataset), split_path=args.out_domain_split)
        metrics, predictions = evaluate_labeled(rows, predicted_cells, "out_domain")
        report["domains"]["out_domain"] = metrics
        write_jsonl(output_dir / "out_domain_predictions.jsonl", predictions)
    elif args.out_domain_prompts:
        metrics, predictions = predict_unlabeled_prompts(args.out_domain_prompts, predicted_cells, "out_domain", limit=args.out_domain_limit)
        report["domains"]["out_domain"] = metrics
        write_jsonl(output_dir / "out_domain_predictions.jsonl", predictions)
    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Offline benchmark for ASCR selector baselines.")
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output-dir", default="outputs/selector_benchmarks/cell_prior_qwen37")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--in-domain-dataset", default=None)
    parser.add_argument("--in-domain-split", default=None)
    parser.add_argument("--out-domain-dataset", default=None)
    parser.add_argument("--out-domain-split", default=None)
    parser.add_argument("--out-domain-prompts", default=None)
    parser.add_argument("--out-domain-limit", type=int, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_selector_benchmark(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Summarize results from all three benchmarks (DPG-Bench, GenAI-Bench, DSG-1k)
across all three models (ShowO50, ASCR50, BAGEL-7B-MoT).

Usage:
  python scripts/benchmark/summarize_bench3.py \
    --eval-root outputs/bench3_eval \
    --output outputs/bench3_summary.json

Expected directory layout under --eval-root:
  dpg_showo/scores_by_category.json
  dpg_ascr/scores_by_category.json
  dpg_bagel/scores_by_category.json
  dsg_showo/scores_by_category.json
  dsg_ascr/scores_by_category.json
  dsg_bagel/scores_by_category.json
  genai_showo/scores.json   (+ scores_by_skill.json)
  genai_ascr/scores.json
  genai_bagel/scores.json
"""

import argparse
import json
from pathlib import Path


MODELS = ["showo", "ascr", "bagel"]
MODEL_LABELS = {"showo": "ShowO50", "ascr": "ASCR50", "bagel": "BAGEL-7B-MoT"}


def load_csv_bench(root: Path, bench: str) -> dict:
    """Load DPG or DSG results for all 3 models."""
    results = {}
    for m in MODELS:
        p = root / f"{bench}_{m}" / "scores_by_category.json"
        if p.exists():
            results[m] = json.loads(p.read_text(encoding="utf-8"))
        else:
            results[m] = None
    return results


def load_genai_bench(root: Path) -> dict:
    results = {}
    for m in MODELS:
        p = root / f"genai_{m}" / "scores.json"
        sp = root / f"genai_{m}" / "scores_by_skill.json"
        if p.exists():
            results[m] = json.loads(p.read_text(encoding="utf-8"))
            if sp.exists():
                results[m]["skills"] = json.loads(sp.read_text(encoding="utf-8"))
        else:
            results[m] = None
    return results


def fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{v*100:.2f}%"


def print_csv_bench_table(bench_name: str, data: dict):
    """Print a comparison table for a CSV-format benchmark."""
    print(f"\n{'='*60}")
    print(f"  {bench_name}")
    print(f"{'='*60}")

    # Collect all categories across models
    all_cats = set()
    for m in MODELS:
        if data[m]:
            all_cats.update(data[m].get("categories", {}).keys())

    header = f"{'Category':<22}" + "".join(f"{MODEL_LABELS[m]:>16}" for m in MODELS)
    print(header)
    print("-" * (22 + 16 * len(MODELS)))

    # Overall first
    row = f"{'OVERALL':<22}"
    for m in MODELS:
        v = data[m]["overall"] if data[m] else None
        row += f"{fmt(v):>16}"
    print(row)

    # Per-category
    for cat in sorted(all_cats):
        row = f"{cat:<22}"
        for m in MODELS:
            v = None
            if data[m] and "categories" in data[m]:
                cat_d = data[m]["categories"].get(cat)
                if cat_d:
                    v = cat_d.get("score")
            row += f"{fmt(v):>16}"
        print(row)


def print_genai_table(data: dict):
    print(f"\n{'='*60}")
    print("  GenAI-Bench")
    print(f"{'='*60}")

    header = f"{'Category':<22}" + "".join(f"{MODEL_LABELS[m]:>16}" for m in MODELS)
    print(header)
    print("-" * (22 + 16 * len(MODELS)))

    row = f"{'OVERALL':<22}"
    for m in MODELS:
        v = data[m]["overall"] if data[m] else None
        row += f"{fmt(v):>16}"
    print(row)

    # Basic skills
    all_basic = set()
    all_adv = set()
    for m in MODELS:
        if data[m] and "skills" in data[m]:
            all_basic.update(data[m]["skills"].get("basic_skills", {}).keys())
            all_adv.update(data[m]["skills"].get("advanced_skills", {}).keys())

    if all_basic:
        print("\n  Basic skills:")
        for skill in sorted(all_basic):
            row = f"  {skill:<20}"
            for m in MODELS:
                v = None
                if data[m] and "skills" in data[m]:
                    sd = data[m]["skills"]["basic_skills"].get(skill)
                    if sd:
                        v = sd.get("score")
                row += f"{fmt(v):>16}"
            print(row)

    if all_adv:
        print("\n  Advanced skills:")
        for skill in sorted(all_adv):
            row = f"  {skill:<20}"
            for m in MODELS:
                v = None
                if data[m] and "skills" in data[m]:
                    sd = data[m]["skills"]["advanced_skills"].get(skill)
                    if sd:
                        v = sd.get("score")
                row += f"{fmt(v):>16}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize bench3 results across all models.")
    parser.add_argument("--eval-root", default="outputs/bench3_eval")
    parser.add_argument("--output", default="outputs/bench3_summary.json")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)

    dpg_data = load_csv_bench(eval_root, "dpg")
    dsg_data = load_csv_bench(eval_root, "dsg")
    genai_data = load_genai_bench(eval_root)

    print_csv_bench_table("DPG-Bench (1065 prompts, ~13.5 Q/prompt, GPT-5.5 VQA)", dpg_data)
    print_csv_bench_table("DSG-1k (1060 prompts, ~8 Q/prompt, GPT-5.5 VQA)", dsg_data)
    print_genai_table(genai_data)

    # Build summary JSON
    summary = {
        "models": {m: MODEL_LABELS[m] for m in MODELS},
        "dpg_bench": {m: ({"overall": dpg_data[m]["overall"], "categories": dpg_data[m]["categories"]}
                          if dpg_data[m] else None) for m in MODELS},
        "dsg1k": {m: ({"overall": dsg_data[m]["overall"], "categories": dsg_data[m]["categories"]}
                      if dsg_data[m] else None) for m in MODELS},
        "genai_bench": {m: ({"overall": genai_data[m]["overall"]}
                            if genai_data[m] else None) for m in MODELS},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Summary saved to {output_path}")


if __name__ == "__main__":
    main()

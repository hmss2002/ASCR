#!/usr/bin/env python3
"""Build a side-by-side GenEval summary across multiple models.

Each model contributes one results.jsonl produced by scripts/judge/evaluate_geneval_owlvit.py
(or the merged file from jobs/benchmarks/stage1_geneval_score_single.sbatch).

Usage:
  python scripts/benchmark/build_geneval_3way_summary.py \
      --model ShowO50=path/to/score_ShowO50.jsonl \
      --model ASCR50=path/to/score_ASCR50.jsonl \
      --model BAGEL=path/to/score_BAGEL.jsonl \
      --output path/to/geneval_3way_summary.md
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

def load(path):
    return pd.read_json(path, orient="records", lines=True)

def summarize(df):
    overall_img  = float(df["correct"].mean())
    overall_prom = float(df.groupby("metadata")["correct"].any().mean())
    per_tag = {}
    for tag, sub in df.groupby("tag", sort=False):
        per_tag[str(tag)] = {
            "n_images": int(len(sub)),
            "n_prompts": int(sub.groupby("metadata").ngroups),
            "image_acc": float(sub["correct"].mean()),
            "prompt_acc": float(sub.groupby("metadata")["correct"].any().mean()),
        }
    return {
        "n_images": int(len(df)),
        "n_prompts": int(df.groupby("metadata").ngroups),
        "overall_image_acc": overall_img,
        "overall_prompt_acc": overall_prom,
        "per_tag": per_tag,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help='LABEL=PATH, repeatable')
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    models = {}
    for entry in args.model:
        if "=" not in entry:
            sys.exit(f"--model expects LABEL=PATH, got: {entry}")
        label, path = entry.split("=", 1)
        models[label.strip()] = Path(path.strip())

    summaries = {label: summarize(load(path)) for label, path in models.items()}

    # Collect union of tags in first-seen order
    tags = []
    for s in summaries.values():
        for t in s["per_tag"]:
            if t not in tags:
                tags.append(t)

    labels = list(summaries.keys())
    lines = []
    lines.append(f"# GenEval 3-way summary")
    lines.append("")
    lines.append("Per-prompt accuracy (any-sample-correct). N = #prompts per task.")
    lines.append("")
    header = "| Task | N | " + " | ".join(labels) + " |"
    sep    = "|---|---:|" + "|".join(["---:"]*len(labels)) + "|"
    lines.append(header)
    lines.append(sep)
    for tag in tags:
        ns = [summaries[l]["per_tag"].get(tag, {}).get("n_prompts", 0) for l in labels]
        n  = max(ns) if ns else 0
        cells = []
        for l in labels:
            v = summaries[l]["per_tag"].get(tag, {}).get("prompt_acc")
            cells.append(f"{v:.2%}" if v is not None else "—")
        lines.append(f"| {tag} | {n} | " + " | ".join(cells) + " |")
    overall_ns = [summaries[l]["n_prompts"] for l in labels]
    overall_n  = max(overall_ns) if overall_ns else 0
    overall_cells = [f"{summaries[l]['overall_prompt_acc']:.2%}" for l in labels]
    lines.append(f"| **Overall** | **{overall_n}** | " + " | ".join(f"**{c}**" for c in overall_cells) + " |")
    lines.append("")
    lines.append("Per-image accuracy (raw):")
    lines.append("")
    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---:"]*len(labels)) + "|")
    lines.append("| Overall image acc | " + " | ".join(f"{summaries[l]['overall_image_acc']:.2%}" for l in labels) + " |")
    lines.append("| #images | " + " | ".join(str(summaries[l]['n_images']) for l in labels) + " |")
    lines.append("")
    lines.append("Sources:")
    for l, p in models.items():
        lines.append(f"- {l}: `{p}`")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Also dump machine-readable
    out.with_suffix(".json").write_text(json.dumps({"models": {l: str(p) for l, p in models.items()},
                                                    "summaries": summaries}, indent=2), encoding="utf-8")
    print("Wrote", out)
    print("Wrote", out.with_suffix(".json"))

if __name__ == "__main__":
    main()

"""Sample Stage-3 self-corruption prompts without holdout contamination."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalise(text):
    return " ".join(str(text).strip().lower().split())


def read_prompt_file(path):
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            prompts.append(text)
    return prompts


def prompt_bucket(prompt):
    words = prompt.split()
    lowered = prompt.lower()
    score = len(words)
    score += sum(2 for token in ("left", "right", "above", "below", "behind", "front", "between") if token in lowered)
    score += sum(1 for token in ("and", "with", "while", "near") if token in lowered)
    if score <= 8:
        return "simple"
    if score <= 18:
        return "medium"
    return "complex"


def sample_prompts(sources, count, holdout=None, seed=0, stratify="complexity"):
    holdout_norm = set()
    for path in holdout or []:
        holdout_norm.update(_normalise(prompt) for prompt in read_prompt_file(path))
    seen = set()
    candidates = []
    for source in sources:
        for prompt in read_prompt_file(source):
            norm = _normalise(prompt)
            if not norm or norm in seen or norm in holdout_norm:
                continue
            seen.add(norm)
            candidates.append({"prompt": prompt, "source": str(source), "bucket": prompt_bucket(prompt)})
    rng = random.Random(seed)
    if stratify != "complexity":
        rng.shuffle(candidates)
        return candidates[: int(count)], candidates
    by_bucket = defaultdict(list)
    for row in candidates:
        by_bucket[row["bucket"]].append(row)
    for rows in by_bucket.values():
        rng.shuffle(rows)
    buckets = ["simple", "medium", "complex"]
    selected = []
    while len(selected) < int(count) and any(by_bucket.values()):
        for bucket in buckets:
            if by_bucket[bucket] and len(selected) < int(count):
                selected.append(by_bucket[bucket].pop())
    return selected, candidates


def build_parser():
    parser = argparse.ArgumentParser(description="Sample Stage-3 prompts with holdout decontamination.")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--stratify", choices=["complexity", "none"], default="complexity")
    parser.add_argument("--holdout", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected, candidates = sample_prompts(
        args.sources,
        args.count,
        holdout=args.holdout,
        seed=args.seed,
        stratify=args.stratify,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(row["prompt"] for row in selected) + "\n", encoding="utf-8")
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "ascr.stage3.sample_prompts.v1",
        "created_at_utc": created_at_utc(),
        "sources": [str(path) for path in args.sources],
        "holdout": [str(path) for path in args.holdout or []],
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "count_requested": int(args.count),
        "stratify": args.stratify,
        "seed": int(args.seed),
        "output": str(output),
        "bucket_counts": {bucket: sum(1 for row in selected if row["bucket"] == bucket) for bucket in ("simple", "medium", "complex")},
        "selected": selected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "selected_count": len(selected)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


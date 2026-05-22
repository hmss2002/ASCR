#!/usr/bin/env python3
"""
GPT-5.5 per-image scoring for GenEval 553-prompt benchmark.

Replaces OWLViT detector-based scoring with GPT-5.5 visual evaluation.
Reads existing result jsonl files (which contain image paths + metadata)
and asks GPT-5.5 to score each image against the prompt requirements.

Usage:
  # Score ShowO50 baseline:
  python scripts/judge_geneval_gpt.py \
      outputs/geneval_parallel_20260522_120250/results_baseline.jsonl \
      --model-name ShowO50 \
      --output outputs/geneval_parallel_20260522_120250/gpt_scores_showo.json \
      --workers 20

  # Score ASCR50:
  python scripts/judge_geneval_gpt.py \
      outputs/geneval_parallel_20260522_120250/results_ascr.jsonl \
      --model-name ASCR50 \
      --output outputs/geneval_parallel_20260522_120250/gpt_scores_ascr.json \
      --workers 20

  # Score BAGEL (uses suite.json format):
  python scripts/judge_geneval_gpt.py \
      outputs/geneval_bagel_68762_20260521_175812/suite.json \
      --model-name BAGEL \
      --bagel-tag-ref outputs/geneval_parallel_20260522_120250/results_baseline.jsonl \
      --output outputs/geneval_bagel_68762_20260521_175812/gpt_scores_bagel.json \
      --workers 20

  # Compare all three (prints table after all three score files exist):
  python scripts/judge_geneval_gpt.py --compare \
      --showo outputs/geneval_parallel_20260522_120250/gpt_scores_showo.json \
      --ascr  outputs/geneval_parallel_20260522_120250/gpt_scores_ascr.json \
      --bagel outputs/geneval_bagel_68762_20260521_175812/gpt_scores_bagel.json

Environment:
  OFOX_API_KEY  — API key for ofox.ai proxy (required)
"""

import argparse
import base64
import io
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import re

OFOX_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_MODEL = "openai/gpt-5.5"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

GENEVAL_TASKS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]

TASK_INSTRUCTIONS = {
    "single_object": "Check: is the specified object clearly present in the image?",
    "two_object": "Check: are BOTH specified objects clearly present in the image?",
    "counting": "Check: is the EXACT specified number of objects present? Count carefully.",
    "colors": "Check: do the specified objects have the correct colors described?",
    "position": "Check: are the objects in the correct spatial relationship (left/right/above/below/in front of/behind)?",
    "color_attr": "Check: do the specified objects have the correct color attributes bound to the correct objects?",
}


def resolve_path(value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path.cwd() / path


def image_to_base64(image_path):
    path = Path(image_path)
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        from PIL import Image
        buf = io.BytesIO()
        with Image.open(path) as img:
            img.convert("RGB").save(buf, format="JPEG", quality=90)
        return "image/jpeg", base64.b64encode(buf.getvalue()).decode()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp"}.get(suffix, "image/jpeg")
    return mime, base64.b64encode(path.read_bytes()).decode()


def load_jsonl_results(path):
    """Load ShowO/ASCR results.jsonl — one JSON object per line."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def load_bagel_suite(suite_path, tag_ref_path=None):
    """Load BAGEL suite.json, optionally adding tags from a reference ShowO jsonl."""
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    results = suite.get("results", [])

    # Build prompt→tag mapping from reference if provided
    prompt_to_tag = {}
    prompt_to_metadata = {}
    if tag_ref_path:
        for item in load_jsonl_results(tag_ref_path):
            prompt = item.get("prompt", "").strip()
            if prompt:
                prompt_to_tag[prompt] = item.get("tag", "unknown")
                prompt_to_metadata[prompt] = item.get("metadata", "")

    items = []
    for r in results:
        prompt = r.get("prompt", "").strip()
        image_path = r.get("bagel_image") or r.get("image") or r.get("image_path")
        tag = r.get("tag") or prompt_to_tag.get(prompt, "unknown")
        metadata = prompt_to_metadata.get(prompt, "")
        items.append({
            "filename": str(image_path) if image_path else None,
            "prompt": prompt,
            "tag": tag,
            "metadata": metadata,
        })
    return items


def build_score_message(item):
    """Build GPT message for scoring a single image."""
    prompt = item["prompt"]
    tag = item["tag"]
    metadata_str = item.get("metadata", "")

    # Parse metadata if it's a JSON string
    try:
        meta = json.loads(metadata_str) if isinstance(metadata_str, str) and metadata_str else {}
    except Exception:
        meta = {}

    # Build requirement description from metadata
    requirements = ""
    if meta.get("include"):
        parts = []
        for obj in meta["include"]:
            cls = obj.get("class", "?")
            count = obj.get("count", 1)
            color = obj.get("color", "")
            if color:
                parts.append(f"{count}× {color} {cls}")
            else:
                parts.append(f"{count}× {cls}")
        requirements = "Required objects: " + ", ".join(parts) + "."
    if meta.get("relation"):
        requirements += f" Spatial relation: {meta['relation']}."

    task_hint = TASK_INSTRUCTIONS.get(tag, "Check if the image accurately depicts the prompt.")
    image_path = resolve_path(item["filename"])
    mime, b64 = image_to_base64(image_path)

    return [
        {
            "role": "system",
            "content": (
                "You are a precise text-to-image evaluation judge. "
                "You check whether generated images correctly depict what is described in the prompt. "
                "Be strict but fair. Focus only on semantic accuracy (what is depicted), "
                "not on artistic quality or photorealism."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                },
                {
                    "type": "text",
                    "text": (
                        f'Prompt: "{prompt}"\n'
                        f'Task category: {tag}\n'
                        + (f'{requirements}\n' if requirements else "")
                        + f'{task_hint}\n\n'
                        'Return exactly one JSON object:\n'
                        '{"correct": true|false, "confidence": 0.0-1.0, "reason": "one sentence"}'
                    ),
                },
            ],
        },
    ]


def extract_json_score(text):
    """Extract correct/confidence/reason from GPT response."""
    match = re.search(r'\{[^{}]*"correct"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    text_lower = text.lower()
    correct = True if '"correct": true' in text_lower or '"correct":true' in text_lower else \
              False if '"correct": false' in text_lower or '"correct":false' in text_lower else None
    if correct is not None:
        return {"correct": correct, "confidence": 0.5, "reason": "extracted from partial response"}
    raise ValueError(f"Could not extract JSON from: {text[:300]}")


def call_gpt(client, model, messages, max_tokens=200):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0.0
            )
            return resp.choices[0].message.content
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise


def score_one(client, model, item, idx):
    """Score a single image. Returns (idx, result_dict)."""
    try:
        messages = build_score_message(item)
        raw = call_gpt(client, model, messages)
        payload = extract_json_score(raw)
        correct = bool(payload.get("correct", False))
        confidence = float(payload.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return idx, {
            "status": "pass",
            "filename": item["filename"],
            "prompt": item["prompt"],
            "tag": item["tag"],
            "correct": correct,
            "confidence": confidence,
            "reason": str(payload.get("reason", ""))[:300],
            "raw_text": raw,
        }
    except Exception as exc:
        return idx, {
            "status": "error",
            "filename": item["filename"],
            "prompt": item["prompt"],
            "tag": item["tag"],
            "correct": False,
            "confidence": 0.0,
            "reason": str(exc)[:300],
        }


def run_score(args):
    """Score all images in the input file."""
    import openai

    api_key = os.environ.get("OFOX_API_KEY")
    if not api_key:
        raise EnvironmentError("OFOX_API_KEY environment variable is not set")
    client = openai.OpenAI(base_url=OFOX_BASE_URL, api_key=api_key)

    input_path = Path(args.input_path)
    model_name = args.model_name or input_path.stem
    model = args.model
    workers = args.workers

    # Load items
    if input_path.suffix == ".jsonl":
        items = load_jsonl_results(input_path)
    else:
        items = load_bagel_suite(input_path, getattr(args, "bagel_tag_ref", None))

    if args.limit:
        items = items[:args.limit]

    output_path = Path(args.output) if args.output else input_path.with_name(f"gpt_scores_{model_name.lower()}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "event": "start", "model_name": model_name, "model": model,
        "total": len(items), "workers": workers
    }), flush=True)

    lock = threading.Lock()
    results_map = {}

    def process(item_idx):
        idx, item = item_idx
        print(json.dumps({"event": "score_start", "index": idx, "prompt": item["prompt"][:60]}), flush=True)
        idx, result = score_one(client, model, item, idx)
        print(json.dumps({"event": "scored", "index": idx, "correct": result["correct"],
                          "tag": result["tag"], "confidence": result["confidence"]}), flush=True)
        return idx, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, (i, item)): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx, result = future.result()
            with lock:
                results_map[idx] = result

    scored = [results_map[i] for i in sorted(results_map)]

    # Per-task accuracy
    task_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in scored:
        tag = r.get("tag", "unknown")
        task_counts[tag]["total"] += 1
        if r.get("correct"):
            task_counts[tag]["correct"] += 1

    overall_correct = sum(1 for r in scored if r.get("correct"))
    overall_total = len(scored)

    task_accuracy = {}
    for tag in GENEVAL_TASKS:
        tc = task_counts[tag]
        if tc["total"] > 0:
            task_accuracy[tag] = {
                "correct": tc["correct"],
                "total": tc["total"],
                "accuracy": round(tc["correct"] / tc["total"] * 100, 2),
            }

    report = {
        "protocol": "gpt_geneval_per_image_score_v1",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "model_name": model_name,
        "gpt_model": model,
        "total": overall_total,
        "overall_correct": overall_correct,
        "overall_accuracy": round(overall_correct / overall_total * 100, 2) if overall_total > 0 else 0,
        "task_accuracy": task_accuracy,
        "records": scored,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== GPT-5.5 GenEval Results ===")
    print(f"Model: {model_name}")
    for tag in GENEVAL_TASKS:
        if tag in task_accuracy:
            ta = task_accuracy[tag]
            print(f"  {tag:15s}: {ta['accuracy']:6.2f}%  ({ta['correct']}/{ta['total']})")
    print(f"  {'OVERALL':15s}: {report['overall_accuracy']:6.2f}%  ({overall_correct}/{overall_total})")
    print(f"\nSaved to {output_path}")
    return 0


def run_compare(args):
    """Print a 3-way comparison table."""
    results = {}
    for name, path in [("ShowO50", args.showo), ("ASCR50", args.ascr), ("BAGEL", args.bagel)]:
        if path and Path(path).exists():
            d = json.loads(Path(path).read_text())
            results[name] = d

    if not results:
        print("No result files found.")
        return 1

    # Header
    models = list(results.keys())
    header = f"{'Task':15s} | {'N':>5} | " + " | ".join(f"{m:>10}" for m in models)
    print("\n=== GPT-5.5 GenEval 3-Way Comparison ===\n")
    print(header)
    print("-" * len(header))

    for tag in GENEVAL_TASKS:
        row = f"{tag:15s} | "
        n = None
        cols = []
        for m in models:
            ta = results[m].get("task_accuracy", {}).get(tag)
            if ta:
                n = ta["total"]
                cols.append(f"{ta['accuracy']:>9.2f}%")
            else:
                cols.append(f"{'N/A':>10}")
        row += f"{(n or 0):>5} | " + " | ".join(cols)
        print(row)

    print("-" * len(header))
    overall_row = f"{'OVERALL':15s} | {'':>5} | "
    overall_cols = []
    for m in models:
        acc = results[m].get("overall_accuracy")
        n = results[m].get("total")
        correct = results[m].get("overall_correct")
        if acc is not None:
            overall_cols.append(f"{acc:>9.2f}%")
        else:
            overall_cols.append(f"{'N/A':>10}")
    print(overall_row + " | ".join(overall_cols))

    # ASCR-ShowO delta
    if "ShowO50" in results and "ASCR50" in results:
        print("\n=== ASCR − ShowO Delta ===")
        for tag in GENEVAL_TASKS:
            s = results["ShowO50"].get("task_accuracy", {}).get(tag, {}).get("accuracy")
            a = results["ASCR50"].get("task_accuracy", {}).get(tag, {}).get("accuracy")
            if s is not None and a is not None:
                delta = a - s
                print(f"  {tag:15s}: {delta:+.2f} pp")
        s_ov = results["ShowO50"].get("overall_accuracy", 0)
        a_ov = results["ASCR50"].get("overall_accuracy", 0)
        print(f"  {'OVERALL':15s}: {a_ov - s_ov:+.2f} pp")

    if args.output:
        summary = {
            "protocol": "gpt_geneval_3way_comparison_v1",
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "models": {m: {
                "overall_accuracy": results[m].get("overall_accuracy"),
                "task_accuracy": results[m].get("task_accuracy"),
            } for m in models},
        }
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nSaved comparison to {args.output}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="GPT-5.5 per-image GenEval scorer")
    parser.add_argument("input_path", nargs="?", default=None,
                        help="results_*.jsonl (ShowO/ASCR) or suite.json (BAGEL). Omit when using --compare.")
    parser.add_argument("--model-name", dest="model_name", default=None,
                        help="Human label for this model (e.g. ShowO50, ASCR50, BAGEL)")
    parser.add_argument("--bagel-tag-ref", dest="bagel_tag_ref", default=None,
                        help="Path to ShowO results_baseline.jsonl to supply task tags for BAGEL")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"GPT model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent API workers (default 20)")
    parser.add_argument("--limit", type=int, default=None, help="Limit prompts (for testing)")

    # Compare mode
    parser.add_argument("--compare", action="store_true", help="Print 3-way comparison table")
    parser.add_argument("--showo", default=None, help="ShowO50 scores JSON (for --compare)")
    parser.add_argument("--ascr", default=None, help="ASCR50 scores JSON (for --compare)")
    parser.add_argument("--bagel", default=None, help="BAGEL scores JSON (for --compare)")

    args = parser.parse_args(argv)
    if args.compare:
        return run_compare(args)
    if not args.input_path:
        parser.error("input_path is required when not using --compare")
    return run_score(args)


if __name__ == "__main__":
    raise SystemExit(main())

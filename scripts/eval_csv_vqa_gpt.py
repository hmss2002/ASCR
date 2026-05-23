#!/usr/bin/env python3
"""
Evaluate DPG-Bench or DSG-1k using GPT-5.5 as VQA judge.

Both benchmarks share the same CSV format:
  item_id, text, keywords, proposition_id, dependency, category_broad,
  category_detailed, tuple, question_natural_language

Evaluation:
  For each question: send image + question to GPT-5.5, get yes/no answer.
  Apply dependency graph: if a parent proposition is answered 'no', all
  its children are automatically marked 'no' (same as original DPG-Bench eval).

Outputs (in --output-dir):
  answers.json         - raw per-question GPT responses
  scores_by_item.json  - {item_id: {score: float, n_q: int, n_yes: int}}
  scores_by_category.json - per-category averages

Usage:
  python scripts/eval_csv_vqa_gpt.py \
    --csv configs/benchmark_data/dpg_bench.csv \
    --image-map outputs/bench3_image_map.json \
    --model-key showo \
    --output-dir outputs/bench3_eval/dpg_showo \
    --workers 30

  export OFOX_API_KEY=sk-of-...
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


GPT_MODEL = "openai/gpt-5.5"
API_BASE = "https://api.ofox.ai/v1"

MAX_RETRIES = 8
BASE_BACKOFF = 5.0   # seconds
MAX_BACKOFF = 120.0  # seconds


def encode_image(path: str) -> Optional[tuple[str, str]]:
    """Return (base64_string, mime_type) for the image at path, or None if missing.
    Non-JPEG/PNG formats (e.g. PPM) are converted to PNG via Pillow."""
    p = Path(path)
    if not p.exists():
        return None
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return base64.b64encode(p.read_bytes()).decode("utf-8"), "image/jpeg"
    if ext == ".png":
        return base64.b64encode(p.read_bytes()).decode("utf-8"), "image/png"
    # Convert any other format (e.g. .ppm) to PNG in memory
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.open(p).convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def ask_gpt_yes_no(client, item_id: str, question: str, image_b64: str, mime: str = "image/png") -> dict:
    """Send one VQA question to GPT-5.5, return {item_id, question, answer, raw}.
    Retries on rate-limit (429) and transient errors with exponential backoff.
    Raises immediately on 400 bad-request (unsupported image, etc.).
    """
    import openai

    prompt_text = (
        f"{question}\n"
        "Answer with a single word: yes or no."
    )
    backoff = BASE_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }],
                max_tokens=8,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            answer = "yes" if raw.startswith("yes") else "no"
            return {"item_id": item_id, "question": question, "answer": answer, "raw": raw}
        except openai.BadRequestError as e:
            # "unsupported image" from ofox.ai can be a masked rate-limit; retry it
            if "unsupported image" in str(e).lower():
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                raise  # genuine bad request (wrong format, etc.)
        except openai.RateLimitError as e:
            # parse suggested retry-after from message if present
            m = re.search(r"Retry in (\d+)s", str(e))
            wait = float(m.group(1)) + 2.0 if m else backoff
            wait = min(wait, MAX_BACKOFF)
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
    raise RuntimeError(f"ask_gpt_yes_no: exhausted {MAX_RETRIES} retries for {item_id}")


def apply_dependency_graph(questions: list[dict], answers: dict[str, str]) -> dict[str, str]:
    """
    Zero out children whose parent proposition failed.
    questions: list of CSV row dicts
    answers: {proposition_id (str) -> 'yes'|'no'}
    """
    # Build parent map: proposition_id -> parent proposition_id (0 = root)
    parent = {row["proposition_id"]: row["dependency"] for row in questions}
    # Process in order: if parent failed, child = no
    result = dict(answers)
    for row in questions:
        pid = row["proposition_id"]
        dep = row["dependency"]
        if dep and dep != "0" and result.get(dep, "no") == "no":
            result[pid] = "no"
    return result


def evaluate_model(
    csv_path: Path,
    image_map: dict,
    model_key: str,
    output_dir: Path,
    workers: int,
    api_key: str,
    bench_prefix: str,
):
    """Full evaluation pipeline for one model on one CSV benchmark."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=API_BASE)

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    # Group by item_id (preserving original CSV order)
    item_order = []
    items: dict[str, list] = {}
    for row in rows:
        raw_id = row["item_id"]
        full_id = f"{bench_prefix}_{raw_id}"
        if full_id not in items:
            items[full_id] = []
            item_order.append(full_id)
        items[full_id].append(row)

    print(f"[{bench_prefix}/{model_key}] {len(item_order)} items, {len(rows)} questions")

    # Build tasks: (full_id, row, image_b64)
    tasks = []
    skipped = 0
    for full_id in item_order:
        entry = image_map.get(full_id)
        if not entry:
            skipped += 1
            continue
        img_path = entry.get(model_key, "")
        if not img_path:
            skipped += 1
            continue
        img_b64 = encode_image(img_path)
        if img_b64 is None:
            skipped += 1
            continue
        img_b64, mime = img_b64
        for row in items[full_id]:
            tasks.append((full_id, row, img_b64, mime))

    if skipped:
        print(f"[{bench_prefix}/{model_key}] Skipped {skipped} items (missing images)")

    print(f"[{bench_prefix}/{model_key}] Submitting {len(tasks)} GPT queries ({workers} workers)...")

    # Execute in parallel
    raw_answers: dict[str, dict[str, str]] = {}  # {full_id: {prop_id: answer}}

    def do_call(task):
        full_id, row, img_b64, mime = task
        return ask_gpt_yes_no(
            client, full_id,
            row["question_natural_language"],
            img_b64, mime,
        ), row["proposition_id"]

    answers_list = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(do_call, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                result, prop_id = fut.result()
                full_id = result["item_id"]
                if full_id not in raw_answers:
                    raw_answers[full_id] = {}
                raw_answers[full_id][prop_id] = result["answer"]
                answers_list.append({**result, "proposition_id": prop_id})
                done += 1
                if done % 200 == 0:
                    print(f"  ... {done}/{len(tasks)} done")
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)

    print(f"[{bench_prefix}/{model_key}] Got {len(answers_list)} answers")

    # Apply dependency graph and compute per-item scores
    scores_by_item = {}
    for full_id in item_order:
        if full_id not in raw_answers:
            continue
        item_questions = items[full_id]
        corrected = apply_dependency_graph(item_questions, raw_answers[full_id])
        n_yes = sum(1 for v in corrected.values() if v == "yes")
        n_q = len(item_questions)
        row0 = item_questions[0]
        scores_by_item[full_id] = {
            "score": n_yes / n_q if n_q > 0 else 0.0,
            "n_yes": n_yes,
            "n_q": n_q,
            "category_broad": row0.get("category_broad", ""),
            "category_detailed": row0.get("category_detailed", ""),
            "prop_answers": corrected,
        }

    # Per-category aggregation
    scores_by_category = {}
    for v in scores_by_item.values():
        cat = v["category_broad"] or "other"
        if cat not in scores_by_category:
            scores_by_category[cat] = {"n_yes": 0, "n_q": 0, "n_items": 0}
        scores_by_category[cat]["n_yes"] += v["n_yes"]
        scores_by_category[cat]["n_q"] += v["n_q"]
        scores_by_category[cat]["n_items"] += 1

    for cat, d in scores_by_category.items():
        d["score"] = d["n_yes"] / d["n_q"] if d["n_q"] > 0 else 0.0

    overall_yes = sum(v["n_yes"] for v in scores_by_item.values())
    overall_q = sum(v["n_q"] for v in scores_by_item.values())
    overall = overall_yes / overall_q if overall_q > 0 else 0.0

    print(f"\n[{bench_prefix}/{model_key}] Overall score: {overall:.4f} ({overall_yes}/{overall_q})")
    print(f"[{bench_prefix}/{model_key}] Per-category:")
    for cat, d in sorted(scores_by_category.items()):
        print(f"  {cat:20s}: {d['score']:.4f}  ({d['n_yes']}/{d['n_q']})")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "answers.json").write_text(
        json.dumps(answers_list, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "scores_by_item.json").write_text(
        json.dumps(scores_by_item, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "scores_by_category.json").write_text(
        json.dumps({"overall": overall, "categories": scores_by_category}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"[{bench_prefix}/{model_key}] Saved results to {output_dir}")
    return {"overall": overall, "categories": scores_by_category}


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPG-Bench or DSG-1k with GPT-5.5 VQA.")
    parser.add_argument("--csv", required=True, help="DPG-Bench or DSG-1k CSV path")
    parser.add_argument("--image-map", required=True, help="image_map.json from build_bench_image_map.py")
    parser.add_argument("--model-key", required=True, choices=["showo", "ascr", "bagel"],
                        help="Which model to evaluate")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--workers", type=int, default=30, help="Parallel API workers")
    parser.add_argument("--bench-prefix", default=None,
                        help="item_id prefix ('dpg' or 'dsg'); auto-detected from CSV filename if not set")
    args = parser.parse_args()

    api_key = os.environ.get("OFOX_API_KEY", "")
    if not api_key:
        print("ERROR: OFOX_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv)
    bench_prefix = args.bench_prefix
    if bench_prefix is None:
        name = csv_path.stem.lower()
        if "dpg" in name:
            bench_prefix = "dpg"
        elif "dsg" in name:
            bench_prefix = "dsg"
        else:
            bench_prefix = name
    print(f"Benchmark prefix: {bench_prefix}")

    image_map = json.loads(Path(args.image_map).read_text(encoding="utf-8"))

    evaluate_model(
        csv_path=csv_path,
        image_map=image_map,
        model_key=args.model_key,
        output_dir=Path(args.output_dir),
        workers=args.workers,
        api_key=api_key,
        bench_prefix=bench_prefix,
    )


if __name__ == "__main__":
    main()

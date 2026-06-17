#!/usr/bin/env python3
"""
Evaluate GenAI-Bench using GPT-5.5 as binary VQA judge.

For each prompt: send image + "Does this image show '{prompt}'? Answer yes or no."
Aggregates per basic_skills and advanced_skills tags.

Outputs (in --output-dir):
  scores.json            - overall accuracy + per-skill breakdowns
  scores_by_skill.json   - detailed per-skill stats
  answers.json           - raw per-item answers

Usage:
  python scripts/judge/eval_genai_gpt.py \
    --metadata configs/benchmarks/data/genai_bench.jsonl \
    --image-map outputs/bench3_image_map.json \
    --model-key showo \
    --output-dir outputs/bench3_eval/genai_showo \
    --workers 30

  export OFOX_API_KEY=your-ofox-api-key
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


DEFAULT_MODEL = "openai/gpt-5.5"
API_BASE = "https://api.ofox.ai/v1"

MAX_RETRIES = 8
BASE_BACKOFF = 1.0   # seconds (reduced from 5.0 for faster recovery)
MAX_BACKOFF = 120.0


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


def ask_gpt_yes_no(client, item_id: str, prompt: str, image_b64: str, mime: str = "image/png",
                   model: str = DEFAULT_MODEL) -> dict:
    """Send one VQA question to the judge model, return {item_id, prompt, answer, raw}.
    Retries on rate-limit (429) and transient errors with exponential backoff.
    Raises immediately on 400 bad-request (unsupported image, etc.).
    """
    import openai

    question = (
        f"Does this image show the following?\n\"{prompt}\"\n"
        "Answer with a single word: yes or no."
    )
    backoff = BASE_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                        {"type": "text", "text": question},
                    ],
                }],
                max_tokens=200,  # thinking models need ~137 reasoning tokens before output
                temperature=0.0,
            )
            raw = response.choices[0].message.content
            if raw is None or raw.strip() == "":
                # reasoning budget exhausted before producing output; treat as retryable
                raise ValueError("empty response content from thinking model")
            raw = raw.strip().lower()
            answer = "yes" if raw.startswith("yes") else "no"
            return {"item_id": item_id, "prompt": prompt, "answer": answer, "raw": raw}
        except openai.BadRequestError as e:
            # "unsupported image" from ofox.ai can be a masked rate-limit; retry it
            if "unsupported image" in str(e).lower():
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                raise  # genuine bad request
        except openai.RateLimitError as e:
            m = re.search(r"Retry in (\d+)s", str(e))
            wait = float(m.group(1)) + random.uniform(1.0, 20.0) if m else backoff
            wait = min(wait, MAX_BACKOFF)
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  WARN [{item_id} attempt {attempt+1}]: {type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(backoff + random.uniform(0, 5.0))
            backoff = min(backoff * 2, MAX_BACKOFF)
    raise RuntimeError(f"ask_gpt_yes_no: exhausted {MAX_RETRIES} retries for {item_id}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GenAI-Bench with VQA judge.")
    parser.add_argument("--metadata", default="configs/benchmarks/data/genai_bench.jsonl",
                        help="genai_bench.jsonl from prepare_bench_data.py")
    parser.add_argument("--image-map", required=True, help="image_map.json from build_bench_image_map.py")
    parser.add_argument("--model-key", required=True, choices=["showo", "ascr", "bagel"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Judge model to use via ofox.ai (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    api_key = os.environ.get("OFOX_API_KEY", "")
    if not api_key:
        print("ERROR: OFOX_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=API_BASE)
    print(f"[GenAI-Bench/{args.model_key}] Judge model: {args.model}")

    # Load metadata
    meta_lines = Path(args.metadata).read_text(encoding="utf-8").splitlines()
    items = [json.loads(l) for l in meta_lines if l.strip()]
    print(f"[GenAI-Bench/{args.model_key}] {len(items)} items loaded")

    # Load image map
    image_map = json.loads(Path(args.image_map).read_text(encoding="utf-8"))

    # Build tasks
    tasks = []
    skipped = 0
    for item in items:
        item_id = item["item_id"]
        entry = image_map.get(item_id)
        if not entry:
            skipped += 1
            continue
        img_path = entry.get(args.model_key, "")
        if not img_path:
            skipped += 1
            continue
        img_b64 = encode_image(img_path)
        if img_b64 is None:
            skipped += 1
            continue
        img_b64, mime = img_b64
        tasks.append((item_id, item["prompt"], img_b64, mime, item.get("basic_skills", []),
                      item.get("advanced_skills", [])))

    if skipped:
        print(f"[GenAI-Bench/{args.model_key}] Skipped {skipped} items (missing images)")

    # Load checkpoint to resume partial runs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.jsonl"
    done_ids: set[str] = set()
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["item_id"])
                except Exception:
                    pass
        if done_ids:
            print(f"[GenAI-Bench/{args.model_key}] Resuming: {len(done_ids)} items already done")
    tasks = [t for t in tasks if t[0] not in done_ids]

    print(f"[GenAI-Bench/{args.model_key}] Submitting {len(tasks)} queries ({args.workers} workers)...")

    answers = []
    done = 0
    ckpt_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def do_call(task):
            item_id, prompt, img_b64, mime, basic, adv = task
            r = ask_gpt_yes_no(client, item_id, prompt, img_b64, mime, model=args.model)
            r["basic_skills"] = basic
            r["advanced_skills"] = adv
            return r

        futures = {pool.submit(do_call, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                answers.append(r)
                with ckpt_lock:
                    with checkpoint_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done += 1
                if done % 200 == 0:
                    print(f"  ... {done}/{len(tasks)} done")
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)

    print(f"[GenAI-Bench/{args.model_key}] Got {len(answers)} answers")

    # Merge previously-checkpointed answers for scoring
    if done_ids and checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e["item_id"] in done_ids:
                        answers.append(e)
                except Exception:
                    pass

    # Aggregate overall
    n_yes = sum(1 for a in answers if a["answer"] == "yes")
    overall = n_yes / len(answers) if answers else 0.0
    print(f"\n[GenAI-Bench/{args.model_key}] Overall: {overall:.4f} ({n_yes}/{len(answers)})")

    # Per-skill aggregation
    basic_skills: dict = {}
    advanced_skills: dict = {}

    for a in answers:
        correct = 1 if a["answer"] == "yes" else 0
        for skill in a.get("basic_skills", []):
            if skill not in basic_skills:
                basic_skills[skill] = {"n_yes": 0, "n": 0}
            basic_skills[skill]["n_yes"] += correct
            basic_skills[skill]["n"] += 1
        for skill in a.get("advanced_skills", []):
            if skill not in advanced_skills:
                advanced_skills[skill] = {"n_yes": 0, "n": 0}
            advanced_skills[skill]["n_yes"] += correct
            advanced_skills[skill]["n"] += 1

    for d in {**basic_skills, **advanced_skills}.values():
        d["score"] = d["n_yes"] / d["n"] if d["n"] > 0 else 0.0

    print("\n  Basic skills:")
    for skill, d in sorted(basic_skills.items()):
        print(f"    {skill:30s}: {d['score']:.4f} ({d['n_yes']}/{d['n']})")
    print("  Advanced skills:")
    for skill, d in sorted(advanced_skills.items()):
        print(f"    {skill:30s}: {d['score']:.4f} ({d['n_yes']}/{d['n']})")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "scores.json").write_text(json.dumps({
        "overall": overall,
        "n_yes": n_yes,
        "n_total": len(answers),
    }, indent=2), encoding="utf-8")
    (out / "scores_by_skill.json").write_text(json.dumps({
        "basic_skills": basic_skills,
        "advanced_skills": advanced_skills,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[GenAI-Bench/{args.model_key}] Results saved to {out}")


if __name__ == "__main__":
    main()

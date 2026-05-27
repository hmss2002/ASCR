#!/usr/bin/env python3
"""
Gemini 3 Flash Preview single-image clean pass/fail judge for Hard64.

Evaluates each image independently (no A/B comparison) — an external,
unbiased judge unlike Qwen3.5-9B which is also the ASCR loop evaluator.

Usage:
  python scripts/judge_hard64_clean_gemini.py \
      --model-key showo \
      --output outputs/hard64_parallel_20260522_120250/gemini_clean_showo.json \
      --workers 8

  python scripts/judge_hard64_clean_gemini.py \
      --model-key ascr \
      --output outputs/hard64_parallel_20260522_120250/gemini_clean_ascr.json

  python scripts/judge_hard64_clean_gemini.py \
      --model-key bagel \
      --output outputs/hard64_parallel_20260522_120250/gemini_clean_bagel.json

Environment:
  OFOX_API_KEY  — API key for ofox.ai proxy (required)
"""

import argparse
import base64
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


OFOX_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
PASS_THRESHOLD = 0.5
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Source data paths (relative to cwd = project root)
SHOWO_ASCR_JUDGE = "outputs/hard64_parallel_20260522_120250/qwen_clean_judge.json"
BAGEL_JUDGE = "outputs/bagel_t2i_compbench_hard64_8gpu_20260519_202625/qwen_clean_bagel_vs_ascr.json"


def load_items(model_key):
    """Load list of {prompt, image_path} for the given model key."""
    if model_key in ("showo", "ascr"):
        data = json.loads(Path(SHOWO_ASCR_JUDGE).read_text(encoding="utf-8"))
        field = "baseline" if model_key == "showo" else "ascr"
        return [
            {"prompt": r["prompt"], "image": r[field]["image"]}
            for r in data["records"]
        ]
    elif model_key == "bagel":
        data = json.loads(Path(BAGEL_JUDGE).read_text(encoding="utf-8"))
        # In this file: baseline=BAGEL images, ascr=old ASCR images
        return [
            {"prompt": r["prompt"], "image": r["baseline"]["image"]}
            for r in data["records"]
        ]
    else:
        raise ValueError(f"Unknown model key: {model_key!r}. Use showo/ascr/bagel.")


def image_to_base64(image_path):
    path = Path(image_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        from PIL import Image
        import io
        with Image.open(path) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=90)
            data = base64.b64encode(buf.getvalue()).decode()
        return "image/jpeg", data
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode()
    return mime, data


def build_message(prompt, image_path):
    mime, b64 = image_to_base64(image_path)
    return [
        {
            "role": "system",
            "content": (
                "You are a strict text-to-image prompt-following judge. "
                "Evaluate whether the image satisfies the given prompt by checking: "
                "objects present, counts, colors, attributes, spatial relations, and overall semantics. "
                "Be precise and objective."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'Prompt: "{prompt}"\n\nDoes this image satisfy the prompt above?',
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                },
                {
                    "type": "text",
                    "text": (
                        "\nReturn exactly one JSON object with no prose or markdown:\n"
                        '{"matches_prompt": true|false, "score": 0.0-1.0, '
                        '"reason": "one sentence listing what is correct or wrong"}'
                    ),
                },
            ],
        },
    ]


def extract_json(text):
    """Extract JSON object from model response."""
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def call_api(client, model, messages):
    """Call API with retries. Returns raw text or raises."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response content from model")
            return content
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def judge_one(client, model, prompt, image_path):
    """Judge a single image. Returns dict with status, payload, raw_text."""
    messages = build_message(prompt, image_path)
    try:
        raw_text = call_api(client, model, messages)
        payload = extract_json(raw_text)
        matches = bool(payload.get("matches_prompt", False))
        try:
            score = float(payload.get("score", 1.0 if matches else 0.0))
            # Normalize 0–10 scale to 0–1 if needed
            if score > 1.0:
                score = score / 10.0
            score = min(1.0, max(0.0, score))
        except (TypeError, ValueError):
            score = 1.0 if matches else 0.0
        status = "pass" if score >= PASS_THRESHOLD else "fail"
        return {
            "status": status,
            "matches_prompt": matches,
            "score": score,
            "reason": str(payload.get("reason", ""))[:500],
            "raw_text": raw_text,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "matches_prompt": False,
            "score": 0.0,
            "reason": "",
        }


def run(args):
    import openai

    api_key = os.environ.get("OFOX_API_KEY")
    if not api_key:
        raise EnvironmentError("OFOX_API_KEY environment variable is not set")

    client = openai.OpenAI(base_url=OFOX_BASE_URL, api_key=api_key)
    model = args.model
    model_key = args.model_key
    workers = args.workers

    items = load_items(model_key)
    if args.limit:
        items = items[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoint (skip errors so they get retried)
    checkpoint = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            for rec in existing.get("records", []):
                if rec.get("status") not in ("error",):
                    checkpoint[rec["prompt"]] = rec
            print(json.dumps({"event": "checkpoint_loaded", "count": len(checkpoint)}), flush=True)
        except Exception:
            pass

    lock = threading.Lock()
    records_map = dict(checkpoint)
    counts = Counter()
    for rec in checkpoint.values():
        counts[rec["status"]] += 1

    def process_item(item):
        prompt = item["prompt"]
        if prompt in checkpoint:
            return  # Already done
        image = item["image"]
        print(json.dumps({"event": "judge_start", "prompt": prompt[:60]}), flush=True)
        result = judge_one(client, model, prompt, image)
        record = {
            "prompt": prompt,
            "image": image,
            "status": result["status"],
            "matches_prompt": result["matches_prompt"],
            "score": result["score"],
            "reason": result["reason"],
        }
        if result["status"] == "error":
            record["error"] = result.get("error", "unknown")
        with lock:
            records_map[prompt] = record
            counts[result["status"]] += 1
            # Save checkpoint after each result
            _save(output_path, model_key, model, items, records_map, counts, done=False)
        print(json.dumps({"event": "judge_done", "prompt": prompt[:60],
                          "status": result["status"], "score": result["score"]}), flush=True)

    print(json.dumps({
        "event": "start",
        "model_key": model_key,
        "model": model,
        "total": len(items),
        "already_done": len(checkpoint),
        "workers": workers,
    }), flush=True)

    todo = [it for it in items if it["prompt"] not in checkpoint]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_item, it) for it in todo]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

    # Final save
    _save(output_path, model_key, model, items, records_map, counts, done=True)

    pass_count = counts.get("pass", 0)
    total = len(items)
    print(f"\n=== Gemini Clean Pass/Fail: {model_key.upper()} ===")
    print(f"Pass:   {pass_count}/{total} = {pass_count/total*100:.1f}%")
    print(f"Fail:   {counts.get('fail', 0)}/{total}")
    print(f"Errors: {counts.get('error', 0)}/{total}")
    print(f"Output: {output_path}")
    return 0


def _save(output_path, model_key, model, items, records_map, counts, done):
    # Preserve original order
    ordered = [records_map[it["prompt"]] for it in items if it["prompt"] in records_map]
    pass_count = counts.get("pass", 0)
    total = len(items)
    summary = {
        "protocol": "gemini_clean_single_image_judge_v1",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "model_key": model_key,
        "model": model,
        "pass_threshold": PASS_THRESHOLD,
        "prompt_count": total,
        "judged_count": len(records_map),
        "done": done,
        "counts": dict(counts),
        "pass_rate": round(pass_count / total * 100, 2) if total > 0 else None,
        "records": ordered,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Gemini 3 Flash Preview single-image clean pass/fail judge for Hard64"
    )
    parser.add_argument("--model-key", required=True, choices=["showo", "ascr", "bagel"],
                        help="Which model's images to judge")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API workers (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

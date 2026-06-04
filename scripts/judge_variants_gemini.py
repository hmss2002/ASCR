#!/usr/bin/env python3
"""
Gemini 3 Flash Preview judge for the Hard64 direct-vs-coarse variant comparison.

Two modes (login-node only; compute nodes have no internet):

  clean    — single-image pass/fail for one arm's manifest (baseline/coarse/direct).
             Reads --items-file manifest.json (from collect_variant_images.py).
  pairwise — direct vs coarse, same prompt, side by side. Each pair is judged
             TWICE with the images swapped (A/B then B/A) to debias position,
             producing win/loss/tie + a debiased win-rate for "direct".

Environment:
  OFOX_API_KEY  — API key for the ofox.ai proxy (required; env only, never a file).
"""

import argparse
import base64
import io
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


def image_to_base64(image_path):
    path = Path(image_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        from PIL import Image
        with Image.open(path) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=90)
            data = base64.b64encode(buf.getvalue()).decode()
        return "image/jpeg", data
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode()
    return mime, data


def extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def call_api(client, model, messages, max_tokens=1024):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response content from model")
            return content
        except Exception:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


# ----------------------------- clean mode -----------------------------

def build_clean_message(prompt, image_path):
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
                {"type": "text", "text": f'Prompt: "{prompt}"\n\nDoes this image satisfy the prompt above?'},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                {"type": "text", "text": (
                    "\nReturn exactly one JSON object with no prose or markdown:\n"
                    '{"matches_prompt": true|false, "score": 0.0-1.0, '
                    '"reason": "one sentence listing what is correct or wrong"}'
                )},
            ],
        },
    ]


def judge_clean_one(client, model, prompt, image_path):
    try:
        raw_text = call_api(client, model, build_clean_message(prompt, image_path))
        payload = extract_json(raw_text)
        matches = bool(payload.get("matches_prompt", False))
        try:
            score = float(payload.get("score", 1.0 if matches else 0.0))
            if score > 1.0:
                score = score / 10.0
            score = min(1.0, max(0.0, score))
        except (TypeError, ValueError):
            score = 1.0 if matches else 0.0
        status = "pass" if score >= PASS_THRESHOLD else "fail"
        return {"status": status, "matches_prompt": matches, "score": score,
                "reason": str(payload.get("reason", ""))[:500]}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "matches_prompt": False, "score": 0.0, "reason": ""}


def load_manifest_items(items_file, image_field):
    data = json.loads(Path(items_file).read_text(encoding="utf-8"))
    records = data.get("records", [])
    items = []
    for r in records:
        image = r.get(image_field)
        if not image:
            continue
        items.append({"prompt": r["prompt"], "image": image})
    return items


def run_clean(args, client):
    items = load_manifest_items(args.items_file, args.image_field)
    if args.limit:
        items = items[:args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            for rec in existing.get("records", []):
                if rec.get("status") != "error":
                    checkpoint[rec["prompt"]] = rec
        except Exception:
            pass

    lock = threading.Lock()
    records_map = dict(checkpoint)
    counts = Counter()
    for rec in checkpoint.values():
        counts[rec["status"]] += 1

    def save(done):
        ordered = [records_map[it["prompt"]] for it in items if it["prompt"] in records_map]
        pass_count = counts.get("pass", 0)
        total = len(items)
        output_path.write_text(json.dumps({
            "protocol": "gemini_variant_clean_v1",
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "arm_label": args.arm_label,
            "image_field": args.image_field,
            "items_file": str(args.items_file),
            "model": args.model,
            "pass_threshold": PASS_THRESHOLD,
            "prompt_count": total,
            "judged_count": len(records_map),
            "done": done,
            "counts": dict(counts),
            "pass_rate": round(pass_count / total * 100, 2) if total else None,
            "records": ordered,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def process(item):
        prompt = item["prompt"]
        if prompt in checkpoint:
            return
        result = judge_clean_one(client, args.model, prompt, item["image"])
        record = {"prompt": prompt, "image": item["image"], "status": result["status"],
                  "matches_prompt": result["matches_prompt"], "score": result["score"],
                  "reason": result["reason"]}
        if result["status"] == "error":
            record["error"] = result.get("error", "unknown")
        with lock:
            records_map[prompt] = record
            counts[result["status"]] += 1
            save(done=False)

    todo = [it for it in items if it["prompt"] not in checkpoint]
    print(json.dumps({"event": "clean_start", "arm": args.arm_label, "total": len(items),
                      "already_done": len(checkpoint), "todo": len(todo)}), flush=True)
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process, it) for it in todo]
            for f in as_completed(futures):
                f.result()
    save(done=True)
    pass_count = counts.get("pass", 0)
    total = len(items)
    print(f"\n=== Gemini Clean: {args.arm_label} ===")
    print(f"Pass: {pass_count}/{total} = {pass_count/total*100:.1f}%" if total else "no items")
    print(f"Errors: {counts.get('error', 0)}/{total}")
    print(f"Output: {output_path}")
    return 0


# ---------------------------- pairwise mode ----------------------------

def build_pairwise_message(prompt, image_a, image_b):
    mime_a, b64_a = image_to_base64(image_a)
    mime_b, b64_b = image_to_base64(image_b)
    return [
        {
            "role": "system",
            "content": (
                "You are a strict text-to-image judge comparing two candidate images for the same prompt. "
                "Decide which image follows the prompt better (objects, counts, colors, attributes, spatial "
                "relations, semantics). If they are equally good, answer tie. Be objective."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'Prompt: "{prompt}"\n\nImage A:'},
                {"type": "image_url", "image_url": {"url": f"data:{mime_a};base64,{b64_a}", "detail": "high"}},
                {"type": "text", "text": "Image B:"},
                {"type": "image_url", "image_url": {"url": f"data:{mime_b};base64,{b64_b}", "detail": "high"}},
                {"type": "text", "text": (
                    "\nWhich image follows the prompt better? Return exactly one JSON object, no prose:\n"
                    '{"winner": "A"|"B"|"tie", "reason": "one sentence"}'
                )},
            ],
        },
    ]


def judge_pairwise_one(client, model, prompt, image_a, image_b):
    raw_text = call_api(client, model, build_pairwise_message(prompt, image_a, image_b))
    payload = extract_json(raw_text)
    winner = str(payload.get("winner", "tie")).strip().upper()
    if winner not in ("A", "B", "TIE"):
        winner = "TIE"
    return {"winner": winner, "reason": str(payload.get("reason", ""))[:500]}


def index_manifest_by_prompt(items_file):
    data = json.loads(Path(items_file).read_text(encoding="utf-8"))
    out = {}
    for r in data.get("records", []):
        out[r["prompt"]] = r.get("final_image")
    return out


def run_pairwise(args, client):
    # direct manifest is left arm, coarse manifest is right arm
    direct = index_manifest_by_prompt(args.direct_manifest)
    coarse = index_manifest_by_prompt(args.coarse_manifest)
    prompts = [p for p in direct if p in coarse and direct[p] and coarse[p]]
    prompts.sort()
    if args.limit:
        prompts = prompts[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            for rec in existing.get("records", []):
                if rec.get("status") != "error":
                    checkpoint[rec["prompt"]] = rec
        except Exception:
            pass

    lock = threading.Lock()
    records_map = dict(checkpoint)
    counts = Counter()

    def recompute_counts():
        counts.clear()
        for rec in records_map.values():
            counts[rec.get("verdict", "error")] += 1

    recompute_counts()

    def save(done):
        ordered = [records_map[p] for p in prompts if p in records_map]
        n = len([r for r in ordered if r.get("verdict") in ("direct", "coarse", "tie")])
        direct_wins = counts.get("direct", 0)
        coarse_wins = counts.get("coarse", 0)
        ties = counts.get("tie", 0)
        decisive = direct_wins + coarse_wins
        output_path.write_text(json.dumps({
            "protocol": "gemini_variant_pairwise_bidirectional_v1",
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "left_arm": "direct",
            "right_arm": "coarse",
            "direct_manifest": str(args.direct_manifest),
            "coarse_manifest": str(args.coarse_manifest),
            "model": args.model,
            "prompt_count": len(prompts),
            "judged_count": n,
            "done": done,
            "counts": dict(counts),
            "direct_wins": direct_wins,
            "coarse_wins": coarse_wins,
            "ties": ties,
            "direct_win_rate_decisive": round(direct_wins / decisive * 100, 2) if decisive else None,
            "direct_win_rate_all": round(direct_wins / n * 100, 2) if n else None,
            "records": ordered,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def resolve(prompt):
        if prompt in checkpoint:
            return
        d_img = direct[prompt]
        c_img = coarse[prompt]
        try:
            # Pass 1: A=direct, B=coarse
            r1 = judge_pairwise_one(client, args.model, prompt, d_img, c_img)
            # Pass 2: A=coarse, B=direct (swapped)
            r2 = judge_pairwise_one(client, args.model, prompt, c_img, d_img)
        except Exception as exc:
            rec = {"prompt": prompt, "status": "error", "verdict": "error", "error": str(exc)}
            with lock:
                records_map[prompt] = rec
                recompute_counts()
                save(done=False)
            return
        # Map each pass to a vote for direct/coarse/tie
        vote1 = {"A": "direct", "B": "coarse", "TIE": "tie"}[r1["winner"]]
        vote2 = {"A": "coarse", "B": "direct", "TIE": "tie"}[r2["winner"]]
        if vote1 == vote2:
            verdict = vote1
        elif "tie" in (vote1, vote2):
            verdict = "tie"
        else:
            verdict = "tie"  # disagreement -> position bias -> tie
        rec = {"prompt": prompt, "status": "ok", "verdict": verdict,
               "vote_direct_left": vote1, "vote_coarse_left": vote2,
               "reason_1": r1["reason"], "reason_2": r2["reason"],
               "direct_image": d_img, "coarse_image": c_img}
        with lock:
            records_map[prompt] = rec
            recompute_counts()
            save(done=False)

    todo = [p for p in prompts if p not in checkpoint]
    print(json.dumps({"event": "pairwise_start", "total": len(prompts),
                      "already_done": len(checkpoint), "todo": len(todo)}), flush=True)
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(resolve, p) for p in todo]
            for f in as_completed(futures):
                f.result()
    save(done=True)
    direct_wins = counts.get("direct", 0)
    coarse_wins = counts.get("coarse", 0)
    ties = counts.get("tie", 0)
    decisive = direct_wins + coarse_wins
    print("\n=== Gemini Pairwise (bidirectional, debiased) ===")
    print(f"direct wins: {direct_wins}  coarse wins: {coarse_wins}  ties: {ties}")
    if decisive:
        print(f"direct decisive win-rate: {direct_wins/decisive*100:.1f}%")
    print(f"Output: {output_path}")
    return 0


def make_client():
    import openai
    api_key = os.environ.get("OFOX_API_KEY")
    if not api_key:
        raise EnvironmentError("OFOX_API_KEY environment variable is not set")
    return openai.OpenAI(base_url=OFOX_BASE_URL, api_key=api_key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("clean", help="Single-image pass/fail for one arm manifest.")
    pc.add_argument("--items-file", required=True, help="manifest.json from collect_variant_images.py")
    pc.add_argument("--image-field", default="final_image", choices=["final_image", "baseline_image"],
                    help="Which image to judge from each record.")
    pc.add_argument("--arm-label", required=True, help="Label for reporting (e.g. direct, coarse, baseline).")
    pc.add_argument("--output", required=True)
    pc.add_argument("--model", default=DEFAULT_MODEL)
    pc.add_argument("--workers", type=int, default=8)
    pc.add_argument("--limit", type=int, default=None)

    pp = sub.add_parser("pairwise", help="Bidirectional direct-vs-coarse comparison.")
    pp.add_argument("--direct-manifest", required=True)
    pp.add_argument("--coarse-manifest", required=True)
    pp.add_argument("--output", required=True)
    pp.add_argument("--model", default=DEFAULT_MODEL)
    pp.add_argument("--workers", type=int, default=8)
    pp.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    client = make_client()
    if args.mode == "clean":
        return run_clean(args, client)
    return run_pairwise(args, client)


if __name__ == "__main__":
    raise SystemExit(main())

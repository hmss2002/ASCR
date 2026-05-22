#!/usr/bin/env python3
"""
GPT-5.5 A/B pairwise judge for Hard64 ShowO-vs-ASCR comparison.

Uses the ofox.ai proxy (OpenAI-compatible) with two separate image tokens
instead of a side-by-side composite, eliminating LEFT/RIGHT position bias.

Usage:
  # Forward direction (A=baseline/ShowO, B=ASCR):
  python scripts/judge_hard64_pairwise_gpt.py \
      outputs/hard64_parallel_20260522_120250/suite.json \
      --output outputs/hard64_parallel_20260522_120250/gpt_pairwise_fwd.json

  # Swap direction (A=ASCR, B=baseline/ShowO):
  python scripts/judge_hard64_pairwise_gpt.py \
      outputs/hard64_parallel_20260522_120250/suite.json \
      --swap \
      --output outputs/hard64_parallel_20260522_120250/gpt_pairwise_swap.json

  # Merge both and compute debiased win rates:
  python scripts/judge_hard64_pairwise_gpt.py --merge \
      --fwd outputs/hard64_parallel_20260522_120250/gpt_pairwise_fwd.json \
      --swap outputs/hard64_parallel_20260522_120250/gpt_pairwise_swap.json

Environment:
  OFOX_API_KEY  — API key for ofox.ai proxy (required)
"""

import argparse
import base64
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


OFOX_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_MODEL = "openai/gpt-5.5"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds between retries


def resolve_path(value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path.cwd() / path


def load_comparison_results(input_path):
    path = Path(input_path)
    if path.is_dir():
        suite_path = path / "suite.json"
        if suite_path.exists():
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            return suite.get("results", []), suite_path
        # Fallback: find deepest suite.json
        candidates = sorted(path.rglob("suite.json"), key=lambda p: p.stat().st_mtime)
        if candidates:
            suite = json.loads(candidates[-1].read_text(encoding="utf-8"))
            return suite.get("results", []), candidates[-1]
        raise FileNotFoundError(f"No suite.json found in {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "suite.json" or "results" in payload:
        return payload.get("results", []), path
    payload.setdefault("result_path", str(path))
    return [payload], path


def image_to_base64(image_path):
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".ppm": "image/ppm", ".webp": "image/webp"}.get(suffix, "image/jpeg")
    # PPM is not web-safe — convert to JPEG in memory via PIL
    if suffix == ".ppm":
        from PIL import Image
        import io
        with Image.open(path) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=90)
            data = base64.b64encode(buf.getvalue()).decode()
        return "image/jpeg", data
    data = base64.b64encode(path.read_bytes()).decode()
    return mime, data


def build_ab_message(prompt, img_a_path, img_b_path):
    """Build an OpenAI multi-image message in A/B format (no LEFT/RIGHT spatial language)."""
    mime_a, b64_a = image_to_base64(img_a_path)
    mime_b, b64_b = image_to_base64(img_b_path)
    return [
        {
            "role": "system",
            "content": (
                "You are a strict text-to-image prompt-following evaluator. "
                "You will be shown two images (A and B) generated from the same text prompt. "
                "Evaluate which image better satisfies the prompt by checking: "
                "objects present, object counts, colors and attributes, spatial relations, and overall coherence. "
                "Be precise and objective. Declare a tie only if neither image is materially better."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'Prompt: "{prompt}"\n\nImage A:',
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_a};base64,{b64_a}", "detail": "high"},
                },
                {
                    "type": "text",
                    "text": "Image B:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_b};base64,{b64_b}", "detail": "high"},
                },
                {
                    "type": "text",
                    "text": (
                        '\nWhich image (A or B) better satisfies the prompt above?\n'
                        'Return exactly one JSON object with no prose or markdown:\n'
                        '{"winner": "A"|"B"|"tie", "confidence": 0.0-1.0, '
                        '"reason": "one sentence explaining key difference"}'
                    ),
                },
            ],
        },
    ]


def extract_json_winner(text):
    """Extract winner JSON from model response, with repair fallback."""
    # Try finding a JSON object anywhere in the response
    match = re.search(r'\{[^{}]*"winner"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try the whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Last resort: look for bare A/B/tie
    text_lower = text.lower()
    if '"winner": "a"' in text_lower or '"winner":"a"' in text_lower:
        return {"winner": "A", "confidence": 0.5, "reason": "extracted from partial response"}
    if '"winner": "b"' in text_lower or '"winner":"b"' in text_lower:
        return {"winner": "B", "confidence": 0.5, "reason": "extracted from partial response"}
    raise ValueError(f"Could not extract JSON winner from response: {text[:300]}")


def call_gpt(client, model, messages, max_tokens=256):
    """Call GPT with retry logic."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS - 1:
                print(json.dumps({"event": "retry", "attempt": attempt + 1, "error": str(exc)}), flush=True)
                time.sleep(RETRY_DELAY)
            else:
                raise


def normalize_verdict(winner_letter, swap):
    """
    Convert A/B winner letter to ascr_win / ascr_loss / tie.

    Forward (swap=False): A=baseline(ShowO), B=ASCR
      A wins → baseline wins → ascr_loss
      B wins → ASCR wins → ascr_win

    Swap (swap=True): A=ASCR, B=baseline(ShowO)
      A wins → ASCR wins → ascr_win
      B wins → baseline wins → ascr_loss
    """
    w = winner_letter.strip().upper()
    if w == "TIE":
        return "pairwise_tie"
    if (w == "B" and not swap) or (w == "A" and swap):
        return "ascr_win"
    if (w == "A" and not swap) or (w == "B" and swap):
        return "ascr_loss"
    return "pairwise_tie"


def judge_one(client, model, prompt, img_a, img_b, swap):
    """Judge a single pair. Returns dict with status, payload, raw_text."""
    messages = build_ab_message(prompt, img_a, img_b)
    try:
        raw_text = call_gpt(client, model, messages)
        payload = extract_json_winner(raw_text)
        winner_letter = str(payload.get("winner", "tie")).upper()
        if winner_letter not in ("A", "B", "TIE"):
            winner_letter = "TIE"
        verdict = normalize_verdict(winner_letter, swap)
        return {
            "status": "pass",
            "winner_letter": winner_letter,
            "payload": {
                "winner_letter": winner_letter,
                "winner": "ascr" if verdict == "ascr_win" else ("baseline" if verdict == "ascr_loss" else "tie"),
                "confidence": float(payload.get("confidence", 0.5)),
                "reason": str(payload.get("reason", ""))[:500],
            },
            "raw_text": raw_text,
            "pairwise_verdict": verdict,
        }
    except Exception as exc:
        return {
            "status": "abstain",
            "error": str(exc),
            "pairwise_verdict": "judge_abstain",
        }


def run_judge(args):
    """Run forward or swap pairwise evaluation."""
    import openai

    api_key = os.environ.get("OFOX_API_KEY")
    if not api_key:
        raise EnvironmentError("OFOX_API_KEY environment variable is not set")

    client = openai.OpenAI(base_url=OFOX_BASE_URL, api_key=api_key)

    results, source_path = load_comparison_results(args.input_path)
    if args.limit:
        results = results[:args.limit]
    if not results:
        raise ValueError("No comparison results found")

    output_path = Path(args.output) if args.output else (
        Path(source_path).parent / ("gpt_pairwise_swap.json" if args.swap else "gpt_pairwise_fwd.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    swap = args.swap
    model = args.model
    records = []
    counts = Counter()

    print(json.dumps({
        "event": "start",
        "mode": "swap" if swap else "forward",
        "model": model,
        "total": len(results),
        "A_is": "ASCR" if swap else "baseline(ShowO)",
        "B_is": "baseline(ShowO)" if swap else "ASCR",
    }), flush=True)

    for idx, result in enumerate(results):
        prompt = result["prompt"]
        baseline_path = resolve_path(result.get("baseline_image"))
        ascr_path = resolve_path(
            result.get("accepted_ascr_image") or result.get("ascr_final_image") or result.get("final_decoded_image")
        )
        if baseline_path is None or ascr_path is None:
            print(json.dumps({"event": "skip", "index": idx, "prompt": prompt, "reason": "missing image paths"}), flush=True)
            continue

        img_a = ascr_path if swap else baseline_path
        img_b = baseline_path if swap else ascr_path

        print(json.dumps({"event": "judge_start", "index": idx, "prompt": prompt}), flush=True)
        judgment = judge_one(client, model, prompt, img_a, img_b, swap)
        verdict = judgment["pairwise_verdict"]
        counts[verdict] += 1

        record = {
            "prompt": prompt,
            "result_path": result.get("result_path"),
            "baseline_image": str(baseline_path),
            "ascr_image": str(ascr_path),
            "A_image": str(img_a),
            "B_image": str(img_b),
            "A_label": "ASCR" if swap else "baseline",
            "B_label": "baseline" if swap else "ASCR",
            "judgment": judgment,
            "pairwise_verdict": verdict,
        }
        records.append(record)
        print(json.dumps({"event": "judged", "index": idx, "verdict": verdict,
                          "winner_letter": judgment.get("winner_letter", "?"),
                          "confidence": judgment.get("payload", {}).get("confidence")}), flush=True)

    report = {
        "protocol": "gpt_ab_pairwise_judge_v1",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_path": str(source_path),
        "model": model,
        "mode": "swap" if swap else "forward",
        "A_label": "ASCR" if swap else "baseline(ShowO)",
        "B_label": "baseline(ShowO)" if swap else "ASCR",
        "prompt_count": len(records),
        "counts": dict(counts),
        "records": records,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"event": "done", "output": str(output_path), "counts": dict(counts)}), flush=True)
    print("\n=== Results ===")
    total_decisive = counts.get("ascr_win", 0) + counts.get("ascr_loss", 0)
    if total_decisive > 0:
        ascr_win_rate = counts.get("ascr_win", 0) / total_decisive * 100
        print(f"ASCR wins:     {counts.get('ascr_win', 0)}")
        print(f"Baseline wins: {counts.get('ascr_loss', 0)}")
        print(f"Ties:          {counts.get('pairwise_tie', 0)}")
        print(f"Abstains:      {counts.get('judge_abstain', 0)}")
        print(f"ASCR win rate (decisive only): {ascr_win_rate:.1f}%")
    return 0


def run_merge(args):
    """Merge fwd + swap results and compute debiased win rates."""
    fwd = json.loads(Path(args.fwd).read_text())
    swap = json.loads(Path(args.swap_file).read_text())

    fwd_counts = fwd["counts"]
    swap_counts = swap["counts"]

    # In forward: ascr_win = ASCR wins, ascr_loss = baseline wins
    # In swap:    ascr_win = ASCR wins, ascr_loss = baseline wins (normalized already)
    total_ascr_wins = fwd_counts.get("ascr_win", 0) + swap_counts.get("ascr_win", 0)
    total_base_wins = fwd_counts.get("ascr_loss", 0) + swap_counts.get("ascr_loss", 0)
    total_ties = fwd_counts.get("pairwise_tie", 0) + swap_counts.get("pairwise_tie", 0)
    total_abstains = fwd_counts.get("judge_abstain", 0) + swap_counts.get("judge_abstain", 0)
    total_decisive = total_ascr_wins + total_base_wins

    print("\n=== Debiased GPT-5.5 A/B Pairwise Results (fwd + swap merged) ===")
    print(f"Total prompts:   {fwd['prompt_count']} fwd + {swap['prompt_count']} swap")
    print(f"ASCR wins:       {total_ascr_wins}")
    print(f"Baseline wins:   {total_base_wins}")
    print(f"Ties:            {total_ties}")
    print(f"Abstains:        {total_abstains}")
    if total_decisive > 0:
        ascr_rate = total_ascr_wins / total_decisive * 100
        base_rate = total_base_wins / total_decisive * 100
        print(f"\nDebiased ASCR win rate:     {ascr_rate:.1f}%  ({total_ascr_wins}/{total_decisive})")
        print(f"Debiased baseline win rate: {base_rate:.1f}%  ({total_base_wins}/{total_decisive})")
    else:
        print("No decisive verdicts — all tied or abstained.")

    # Save merged summary
    if args.output:
        summary = {
            "protocol": "gpt_ab_pairwise_debiased_v1",
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "fwd_path": args.fwd,
            "swap_path": args.swap_file,
            "model": fwd.get("model"),
            "fwd_counts": fwd_counts,
            "swap_counts": swap_counts,
            "merged": {
                "ascr_wins": total_ascr_wins,
                "baseline_wins": total_base_wins,
                "ties": total_ties,
                "abstains": total_abstains,
                "decisive": total_decisive,
                "ascr_win_rate": round(total_ascr_wins / total_decisive * 100, 2) if total_decisive > 0 else None,
            },
        }
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nSaved merged summary to {args.output}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="GPT-5.5 A/B pairwise judge for Hard64 ShowO-vs-ASCR")
    parser.add_argument("input_path", nargs="?", default=None, help="suite.json or output dir (omit when using --merge)")
    parser.add_argument("--swap", action="store_true", help="Swap A/B: A=ASCR, B=baseline")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts (for testing)")

    # Merge flags
    parser.add_argument("--merge", action="store_true", help="Merge fwd+swap results and print debiased stats")
    parser.add_argument("--fwd", default=None, help="Forward results JSON (for --merge)")
    parser.add_argument("--swap-file", dest="swap_file", default=None, help="Swap results JSON (for --merge)")

    args = parser.parse_args(argv)
    if args.merge:
        if not args.fwd or not args.swap_file:
            parser.error("--merge requires --fwd and --swap-file")
        return run_merge(args)
    if not args.input_path:
        parser.error("input_path is required when not using --merge")
    return run_judge(args)


if __name__ == "__main__":
    raise SystemExit(main())

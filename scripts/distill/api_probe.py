#!/usr/bin/env python3
"""Probe the OFOX/OpenAI-compatible teacher API without printing secrets."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ascr.distill.api_client import DEFAULT_MODEL, api_settings, build_client, chat_completion_text


def is_empty_content_error(exc):
    return "empty api response content" in str(exc).lower()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check ASCR API teacher connectivity.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--allow-empty-content", action="store_true", help="Treat an empty model response as a non-blocking warning. Useful for qwen routes where tiny probes can false-fail.")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)

    settings = api_settings()
    model = args.model or settings["model"] or DEFAULT_MODEL
    base_url = args.base_url or settings["base_url"]
    try:
        client = build_client(base_url=base_url)
        text = chat_completion_text(
            client,
            [{"role": "user", "content": "Return exactly: OK"}],
            model=model,
            max_tokens=64,
            retries=args.retries,
        )
        payload = {
            "ok": True,
            "blocking": False,
            "base_url": base_url,
            "model": model,
            "response_preview": text[:120],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        empty_content = is_empty_content_error(exc)
        non_blocking = bool(args.allow_empty_content and empty_content)
        payload = {
            "ok": False,
            "blocking": not non_blocking,
            "base_url": base_url,
            "model": model,
            "error_type": "empty_content" if empty_content else exc.__class__.__name__,
            "error": str(exc)[:300],
        }
        if non_blocking:
            payload["warning"] = "empty probe response treated as non-blocking; main teacher run will perform task-level retries"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if non_blocking else 2


if __name__ == "__main__":
    raise SystemExit(main())

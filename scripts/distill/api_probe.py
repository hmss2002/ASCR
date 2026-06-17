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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check ASCR API teacher connectivity.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)

    settings = api_settings()
    model = args.model or settings["model"] or DEFAULT_MODEL
    base_url = args.base_url or settings["base_url"]
    client = build_client(base_url=base_url)
    text = chat_completion_text(
        client,
        [{"role": "user", "content": "Reply with exactly this JSON: {\"ok\": true}"}],
        model=model,
        max_tokens=64,
        retries=2,
    )
    payload = {
        "ok": True,
        "base_url": base_url,
        "model": model,
        "response_preview": text[:120],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ascr.analysis.stage3_token_repair import CLEAN_ROW_SCHEMA, write_json, write_jsonl
from ascr.generators.lumina_native import CODEBOOK_SIZE, IMAGE_TOKEN_OFFSET, LuminaNativeEngine


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_prompts(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _mock_tokens(prompt_index, token_grid_size):
    count = int(token_grid_size) * int(token_grid_size)
    return [IMAGE_TOKEN_OFFSET + ((int(prompt_index) + index) % CODEBOOK_SIZE) for index in range(count)]


def generate_clean_tokens(
    prompts,
    output_dir,
    prompt_offset=0,
    prompt_limit=None,
    seed=0,
    repo_path="third_party/Lumina-DiMOO",
    checkpoint_path="models/lumina-dimoo",
    device="cuda",
    image_size=1024,
    token_grid_size=64,
    mock=False,
):
    output_dir = Path(output_dir)
    token_dir = output_dir / "clean_tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    all_prompts = _read_prompts(prompts)
    start = int(prompt_offset or 0)
    stop = len(all_prompts) if prompt_limit is None else min(len(all_prompts), start + int(prompt_limit))
    selected = list(enumerate(all_prompts[start:stop], start=start))
    engine = None
    if not mock:
        engine = LuminaNativeEngine(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            device=device,
            image_size=image_size,
            token_grid_size=token_grid_size,
        )
    rows = []
    for prompt_index, prompt in selected:
        sample_id = f"clean_p{prompt_index:05d}"
        token_path = token_dir / f"{sample_id}_vq_ids.json"
        sample_seed = int(seed) + int(prompt_index)
        if not token_path.exists():
            tokens = _mock_tokens(prompt_index, token_grid_size) if mock else engine.generate(prompt, seed=sample_seed)
            write_json(token_path, [int(value) for value in tokens])
        rows.append({
            "schema_version": CLEAN_ROW_SCHEMA,
            "created_at_utc": _created_at(),
            "sample_id": sample_id,
            "prompt_index": int(prompt_index),
            "prompt": prompt,
            "clean_vq_ids_path": str(token_path),
            "token_grid_size": int(token_grid_size),
            "image_size": int(image_size),
            "seed": sample_seed,
            "mock": bool(mock),
        })
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(manifest_path, rows)
    summary = {
        "schema_version": "ascr.stage3.clean_vq_token.summary.v1",
        "created_at_utc": _created_at(),
        "prompts": str(prompts),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "prompt_offset": start,
        "prompt_limit": None if prompt_limit is None else int(prompt_limit),
        "row_count": len(rows),
        "token_grid_size": int(token_grid_size),
        "image_size": int(image_size),
        "mock": bool(mock),
    }
    write_json(summary_path, summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Generate clean Lumina VQ token shards for Stage-3 token repair data.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-path", default="third_party/Lumina-DiMOO")
    parser.add_argument("--checkpoint-path", default="models/lumina-dimoo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--token-grid-size", type=int, default=64)
    parser.add_argument("--mock", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = generate_clean_tokens(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

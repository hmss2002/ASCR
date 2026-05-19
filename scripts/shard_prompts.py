import argparse
import json
from pathlib import Path

from ascr.cli.compare_showo_ascr import load_prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split prompt input into contiguous worker shards.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prompt", default="A red cube left of a blue sphere")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--prompt-limit", type=int, default=None)
    args = parser.parse_args(argv)

    prompts = load_prompts(args.prompt, args.prompts_file, args.prompt_limit)
    worker_count = max(1, min(int(args.workers), len(prompts)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    for shard_index in range(worker_count):
        start = len(prompts) * shard_index // worker_count
        end = len(prompts) * (shard_index + 1) // worker_count
        shard_prompts = prompts[start:end]
        shard_path = output_dir / f"shard_{shard_index}.txt"
        shard_path.write_text(chr(10).join(shard_prompts) + chr(10), encoding="utf-8")
        shards.append(
            {
                "index": shard_index,
                "path": str(shard_path),
                "prompt_count": len(shard_prompts),
                "start": start,
                "end": end,
            }
        )

    manifest = {
        "prompt_count": len(prompts),
        "requested_workers": int(args.workers),
        "shard_count": worker_count,
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    (output_dir / "shard_count.txt").write_text(str(worker_count) + chr(10), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

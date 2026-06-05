"""Download MMaDA-8B weights for the ASCR Stage-1 self-evaluation task.

The MAGVIT-v2 image tokenizer (``showlab/magvitv2``) is shared with Show-o and is
expected to already exist at ``models/magvitv2``; only the MMaDA LM is fetched here.

Usage:
    python scripts/setup/download_mmada_models.py \
        --repo Gen-Verse/MMaDA-8B-MixCoT \
        --dest models/mmada-8b-mixcot
"""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download MMaDA-8B weights via huggingface_hub.")
    parser.add_argument("--repo", default="Gen-Verse/MMaDA-8B-MixCoT", help="HuggingFace repo id.")
    parser.add_argument("--dest", default="models/mmada-8b-mixcot", help="Local destination directory.")
    parser.add_argument("--revision", default=None, help="Optional git revision / branch / tag.")
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin.index.json.lock", "*.msgpack", "*.h5", "*.ot"],
    )
    print(f"Downloaded {args.repo} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

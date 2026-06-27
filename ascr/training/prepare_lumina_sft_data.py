import argparse
from datetime import datetime, timezone
import json
import math
import os
import pickle
from pathlib import Path


SYSTEM_PROMPT = (
    "You are the ASCR semantic evaluator for a generated image. Compare the "
    "current image against the original prompt. Return exactly one compact JSON "
    "object. No markdown. No analysis. Do not output a quoted JSON string or "
    "escaped JSON; output the object itself."
)


def read_jsonl(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_target_text(example):
    if example.get("target_json") is not None:
        return json.dumps(example["target_json"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(example.get("target_text", "")).strip()


def _load_image_tokenizer(checkpoint_path, device, image_size):
    import torch
    from diffusers import VQModel
    from PIL import Image
    from utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop

    vqvae = VQModel.from_pretrained(checkpoint_path, subfolder="vqvae").to(device)

    def tokenize(image_path):
        with Image.open(str(image_path)) as opened:
            image = opened.convert("RGB")
        crop_size_list = generate_crop_size_list((int(image_size) // 32) ** 2, 32)
        image = var_center_crop(image, crop_size_list=crop_size_list)
        tokens = encode_img_with_breaks(image, vqvae=vqvae)
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return {
            "input_ids": list(tokens),
            "height": int(image.size[1]),
            "width": int(image.size[0]),
        }

    return tokenize


def _token_payload_from_vq_ids(vq_ids_path, image_size=None, token_grid_size=None):
    vq_ids = json.loads(Path(vq_ids_path).read_text(encoding="utf-8"))
    grid = int(token_grid_size or math.isqrt(len(vq_ids)))
    if grid * grid != len(vq_ids):
        raise ValueError(f"VQ token count is not a square grid: {vq_ids_path}")
    height = width = int(image_size or grid * 16)
    return {
        "input_ids": [int(value) for value in vq_ids],
        "height": height,
        "width": width,
    }


def convert_sft_examples(
    sft_examples,
    output_dir,
    checkpoint_path="models/lumina-dimoo",
    repo_path=None,
    device="cuda",
    image_size=1024,
    limit=None,
    image_tokenizer=None,
):
    if repo_path:
        import sys

        repo = str(Path(repo_path).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
    output_dir = Path(output_dir)
    image_cache_dir = output_dir / "image_tokens"
    image_cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    direct_vq_count = 0
    for index, example in enumerate(read_jsonl(sft_examples)):
        if limit is not None and index >= int(limit):
            break
        raw_image_path = example.get("image_path") or ""
        image_path = Path(raw_image_path) if raw_image_path else None
        raw_vq_ids_path = example.get("vq_ids_path") or example.get("corrupted_vq_ids_path") or ""
        vq_ids_path = Path(raw_vq_ids_path) if raw_vq_ids_path else None
        has_image = bool(image_path and image_path.exists())
        has_vq_ids = bool(vq_ids_path and vq_ids_path.exists())
        if not has_image and not has_vq_ids:
            skipped.append({
                "sample_id": example.get("sample_id"),
                "image_path": str(image_path) if image_path else None,
                "vq_ids_path": str(vq_ids_path) if vq_ids_path else None,
                "reason": "missing_image_or_vq_ids",
            })
            continue
        target_text = compact_target_text(example)
        if not target_text:
            skipped.append({
                "sample_id": example.get("sample_id"),
                "image_path": str(image_path) if image_path else None,
                "vq_ids_path": str(vq_ids_path) if vq_ids_path else None,
                "reason": "missing_target",
            })
            continue
        token_path = image_cache_dir / f"img_{len(rows):04d}.pkl"
        if not token_path.exists():
            with token_path.open("wb") as handle:
                if has_vq_ids:
                    pickle.dump(
                        _token_payload_from_vq_ids(
                            vq_ids_path,
                            image_size=example.get("image_size", image_size),
                            token_grid_size=example.get("token_grid_size"),
                        ),
                        handle,
                    )
                else:
                    if image_tokenizer is None:
                        image_tokenizer = _load_image_tokenizer(checkpoint_path, device, image_size)
                    pickle.dump(image_tokenizer(image_path), handle)
        if has_vq_ids:
            direct_vq_count += 1
        rows.append({
            "sample_id": example.get("sample_id"),
            "user_image": str(token_path).replace("\\", "/"),
            "answer_image": "",
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": example.get("input_text") or f"Original prompt: {example.get('prompt', '')}",
            "answer_text": target_text,
            "source_image": str(image_path).replace("\\", "/") if image_path else None,
            "source_vq_ids": str(vq_ids_path).replace("\\", "/") if vq_ids_path else None,
            "input_mode": "vq_tokens" if has_vq_ids else "image",
        })

    train_path = output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    manifest = {
        "schema_version": "ascr.lumina_sft_data.v2",
        "target_schema": "canonical_semantic_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "sft_examples": str(sft_examples),
        "checkpoint_path": str(checkpoint_path),
        "repo_path": str(repo_path) if repo_path else None,
        "image_size": int(image_size),
        "example_count": len(rows),
        "skipped_count": len(skipped),
        "direct_vq_example_count": direct_vq_count,
        "image_encoded_example_count": len(rows) - direct_vq_count,
        "skipped": skipped,
        "train_jsonl": str(train_path),
        "image_cache_dir": str(image_cache_dir),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Convert ASCR Lumina evaluator SFT examples into Lumina-DiMOO training JSONL.")
    parser.add_argument("--sft-examples", default=os.environ.get("SFT_EXAMPLES", os.environ.get("DATASET", "outputs/stage2_lumina_native/sft_smoke/sft_examples.jsonl")))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs/stage2_lumina_native/lumina_sft_data_v2"))
    parser.add_argument("--checkpoint-path", default=os.environ.get("LUMINA_MODEL_PATH", "models/lumina-dimoo"))
    parser.add_argument("--repo-path", default=os.environ.get("LUMINA_REPO"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = convert_sft_examples(
        args.sft_examples,
        args.output_dir,
        checkpoint_path=args.checkpoint_path,
        repo_path=args.repo_path,
        device=args.device,
        image_size=args.image_size,
        limit=args.limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert ASCR SFT examples to Lumina-DiMOO training format.

Reads sft_examples.jsonl, pre-tokenizes images into pickle files,
and writes a Lumina-compatible JSONL dataset.
"""
import json, os, sys, pickle
from pathlib import Path

# Add Lumina repo to path
LUMINA_REPO = os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO")
sys.path.insert(0, str(Path(LUMINA_REPO).resolve()))

from utils.image_utils import (
    encode_img_with_breaks, calculate_vq_params,
    generate_crop_size_list, var_center_crop,
)
from PIL import Image
from diffusers import VQModel
import torch

SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,
    "eoi": 126350,
}

def tokenize_image(image_path, vqvae, image_size=1024):
    """Encode image to VQ tokens and save as pickle."""
    img = Image.open(image_path).convert("RGB")
    crop_size_list = generate_crop_size_list((image_size // 32) ** 2, 32)
    img = var_center_crop(img, crop_size_list=crop_size_list)
    iw, ih = img.size
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    seq_len, newline_every, token_grid_h, token_grid_w = calculate_vq_params(ih, iw, vae_scale)
    tokens = encode_img_with_breaks(img, vqvae=vqvae)
    return {
        "input_ids": tokens,
        "height": ih,
        "width": iw,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-examples", default="outputs/stage2_lumina_native/sft_smoke/sft_examples.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage2_lumina_native/sft_smoke/lumina_format")
    parser.add_argument("--checkpoint-path", default="models/lumina-dimoo")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = out_dir / "image_tokens"
    image_cache_dir.mkdir(exist_ok=True)

    # Load VQ-VAE once
    print("Loading VQ-VAE...")
    vqvae = VQModel.from_pretrained(args.checkpoint_path, subfolder="vqvae").to("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    with open(args.sft_examples) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            item = json.loads(line)
            image_path = item["image_path"]
            target = item["target_text"]

            # Tokenize image
            pkl_name = f"img_{i:04d}.pkl"
            pkl_path = image_cache_dir / pkl_name
            if not pkl_path.exists():
                try:
                    img_data = tokenize_image(image_path, vqvae)
                    with open(pkl_path, "wb") as pf:
                        pickle.dump(img_data, pf)
                except Exception as e:
                    print(f"WARNING: failed to tokenize {image_path}: {e}")
                    continue

            # Build Lumina Understanding format
            row = {
                "user_image": str(pkl_path),
                "answer_image": "",
                "system_prompt": "You are the ASCR semantic evaluator for a generated image. Compare the current image against the original prompt. Return exactly one compact JSON object. No markdown. No analysis.",
                "user_prompt": f"Original prompt: {item['prompt']}\n\nEvaluate whether the image matches the prompt. Return JSON with has_error, summary, regions, correction_instruction.",
                "answer_text": target,
            }
            rows.append(row)

    # Write JSONL
    jsonl_path = out_dir / "train.jsonl"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Write manifest
    manifest = {
        "schema_version": "ascr.lumina_sft_data.v1",
        "example_count": len(rows),
        "jsonl_path": str(jsonl_path),
        "image_cache_dir": str(image_cache_dir),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done: {len(rows)} examples written to {jsonl_path}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()

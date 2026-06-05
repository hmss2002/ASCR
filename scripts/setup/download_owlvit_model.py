#!/usr/bin/env python3
"""
Download OWLViT model for GenEval independent evaluation.
Run once on the login node (requires internet access).

Usage:
    python scripts/setup/download_owlvit_model.py [--model-id google/owlvit-base-patch32]
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/owlvit-base-patch32",
                        help="HuggingFace model ID to download")
    parser.add_argument("--output-dir", default=None,
                        help="Local directory to save model (default: models/{model_name})")
    args = parser.parse_args()

    model_id = args.model_id
    model_name = model_id.split("/")[-1]
    output_dir = Path(args.output_dir or f"models/{model_name}")

    print(f"Downloading {model_id} to {output_dir} ...", flush=True)

    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(output_dir))
    model.save_pretrained(str(output_dir))

    print(f"Saved OWLViT model to {output_dir}")
    print(f"  Disk usage: {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

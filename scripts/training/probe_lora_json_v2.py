#!/usr/bin/env python3
"""Probe LoRA Lumina JSON compliance - direct path, no LuminaNativeEngine."""
import os, sys, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from PIL import Image
from diffusers import VQModel

LUMINA_REPO = os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO")
sys.path.insert(0, str(Path(LUMINA_REPO).resolve()))
sys.path.insert(0, "/grp01/cds_bdai/JianyuZhang/ASCR")

from model import LLaDAForMultiModalGeneration
from utils.image_utils import encode_img_with_breaks, calculate_vq_params, generate_crop_size_list, var_center_crop, add_break_line
from utils.prompt_utils import generate_multimodal_understanding_prompt
from generators.text_understanding_generator import generate_text_understanding
from ascr.distill.teacher import extract_json_object
from ascr.core.schemas import safe_parse_semantic_evaluation

SP = {"mask":126336,"newline":126084,"answer_start":126354,"answer_end":126355,"boi":126349,"eoi":126350}

def answer_image(model, tok, vqvae, question, image_path, max_new_tokens=384, image_size=512):
    img = Image.open(image_path).convert("RGB")
    crop_list = generate_crop_size_list((image_size//32)**2, 32)
    img = var_center_crop(img, crop_size_list=crop_list)
    iw, ih = img.size
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    _, _, th, tw = calculate_vq_params(ih, iw, vae_scale)
    img_tokens = encode_img_with_breaks(img, vqvae=vqvae)
    img_tokens = add_break_line(img_tokens, th, tw, new_number=SP["newline"])

    input_prompt = generate_multimodal_understanding_prompt(question)
    input_ids = tok(input_prompt)["input_ids"]
    input_token = input_ids[:-1] + img_tokens + input_ids[-1:]
    code_start = len(input_token) + 1
    input_token = input_token + [SP["answer_start"]] + [SP["mask"]] * int(max_new_tokens) + [SP["answer_end"]]
    device = next(model.parameters()).device
    input_ids_t = torch.tensor(input_token, device=device).unsqueeze(0)

    gen_len = int(max_new_tokens)
    block_len = 128
    gen_len = (gen_len // block_len) * block_len
    if gen_len < block_len: gen_len = block_len
    num_blocks = gen_len // block_len
    steps = 64
    if steps % num_blocks != 0:
        steps = (steps // num_blocks) * num_blocks
        if steps < num_blocks: steps = num_blocks

    out = generate_text_understanding(
        model, input_ids_t, steps=steps, gen_length=gen_len, block_length=block_len,
        temperature=0.0, cfg_scale=0.0, remasking="low_confidence", code_start=code_start,
    )
    text = tok.batch_decode(out[:, code_start:-1], skip_special_tokens=True)[0]
    return text.strip()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="models/lumina-dimoo")
    parser.add_argument("--lora_path", default="outputs/stage2_lumina_native/sft_smoke/lora_checkpoint")
    parser.add_argument("--dataset", default="outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl")
    parser.add_argument("--image_root", default="outputs/lumina_qwen_hard64")
    parser.add_argument("--output_dir", default="outputs/stage2_lumina_native/json_probe_lora_v2")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model + LoRA...")
    model = LLaDAForMultiModalGeneration.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.lora_path)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    vqvae = VQModel.from_pretrained(args.base_model, subfolder="vqvae").to(device)
    print("Loaded.")

    with open(args.dataset) as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:args.limit]

    parsed_count = malformed_count = call_error_count = 0
    results = []

    for i, row in enumerate(rows):
        loc = row["localizations"][0]
        image_path = Path(args.image_root) / loc["grid_image"]
        prompt = loc.get("prompt") or row.get("prompt") or ""

        if not Path(image_path).exists():
            print(f"[{i}] SKIP: image not found")
            continue

        # Build evaluator prompt
        question = (
            "You are the ASCR semantic evaluator. Compare the image against the prompt.\n"
            "Return exactly one JSON: {\"has_error\": bool, \"summary\": str, \"regions\": [...], \"correction_instruction\": str}\n"
            f"Prompt: {prompt}"
        )

        try:
            raw_text = answer_image(model, tok, vqvae, question, str(image_path))
            payload = extract_json_object(raw_text)
            parsed = safe_parse_semantic_evaluation(payload, grid_size=4, max_selected_cells=6)
            parsed_count += 1
            status = "parsed"
            error = None
        except Exception as e:
            status = "malformed"
            malformed_count += 1
            error = str(e)[:500]
            raw_text = locals().get("raw_text", "")

        results.append({"idx": i, "prompt": prompt[:100], "status": status, "raw_preview": str(raw_text)[:300], "error": error})
        print(f"[{i}] {status}: {str(raw_text)[:120]}")

    summary = {
        "row_count": len(results), "parsed_count": parsed_count,
        "malformed_count": malformed_count, "call_error_count": call_error_count,
        "parse_rate": parsed_count / max(1, len(results)),
        "lora_path": args.lora_path,
    }
    with open(out_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    with open(out_dir / "probe_rows.jsonl", "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

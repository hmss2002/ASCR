#!/usr/bin/env python3
"""Probe LoRA-finetuned Lumina for JSON compliance."""
import os, sys, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from peft import PeftModel

LUMINA_REPO = os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO")
sys.path.insert(0, str(Path(LUMINA_REPO).resolve()))

from model import LLaDAForMultiModalGeneration
from ascr.evaluators.lumina_native import call_native_answer, native_eval_prompt
from ascr.generators.lumina_native import LuminaNativeEngine
from ascr.distill.teacher import extract_json_object
from ascr.core.schemas import safe_parse_semantic_evaluation

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="models/lumina-dimoo")
    parser.add_argument("--lora_path", default="outputs/stage2_lumina_native/sft_smoke/lora_checkpoint")
    parser.add_argument("--dataset", default="outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl")
    parser.add_argument("--image_root", default="outputs/lumina_qwen_hard64")
    parser.add_argument("--output_dir", default="outputs/stage2_lumina_native/json_probe_lora")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model + LoRA adapter...")
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("LoRA loaded (no merge).")

    # Build engine with merged model
    engine = LuminaNativeEngine(
        checkpoint_path=args.base_model,
        repo_path=os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO"),
        device=device,
        image_size=512,
    )
    engine._model = model
    engine._tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    from diffusers import VQModel
    engine._vqvae = VQModel.from_pretrained(args.base_model, subfolder="vqvae").to(device)
    engine._torch = torch
    # Re-init lumina helpers
    from utils import image_utils, prompt_utils
    from generators.image_generation_generator import generate_image
    engine._lumina = {
        "image_utils": image_utils,
        "prompt_utils": prompt_utils,
        "generate_image": generate_image,
        "templates": prompt_utils.create_prompt_templates(),
    }

    # Load dataset
    with open(args.dataset) as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:args.limit]

    parsed_count = 0
    malformed_count = 0
    call_error_count = 0
    results = []

    for i, row in enumerate(rows):
        loc = row["localizations"][0]
        image_path = Path(args.image_root) / loc["grid_image"]
        prompt = loc.get("prompt") or row.get("prompt") or ""

        if not Path(image_path).exists():
            print(f"[{i}] SKIP: image not found {image_path}")
            continue

        try:
            question = native_eval_prompt(prompt, grid_size=4, max_selected_cells=6)
            raw_text, method = call_native_answer(engine, question, image_path=str(image_path), max_new_tokens=384)
            payload = extract_json_object(raw_text)
            parsed = safe_parse_semantic_evaluation(payload, grid_size=4, max_selected_cells=6)
            parsed_count += 1
            status = "parsed"
            error = None
        except Exception as e:
            status = "call_error" if "method" not in dir() else "malformed"
            if status == "call_error":
                call_error_count += 1
            else:
                malformed_count += 1
            error = str(e)[:500]
            raw_text = locals().get("raw_text", "")
            method = locals().get("method", "")

        results.append({
            "idx": i, "prompt": prompt[:100], "status": status,
            "method": method, "raw_preview": str(raw_text)[:300], "error": error,
        })
        print(f"[{i}] {status}: {str(raw_text)[:100]}")

    summary = {
        "row_count": len(results),
        "parsed_count": parsed_count,
        "malformed_count": malformed_count,
        "call_error_count": call_error_count,
        "parse_rate": parsed_count / max(1, len(results)),
        "lora_path": args.lora_path,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "probe_rows.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

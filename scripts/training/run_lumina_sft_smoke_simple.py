#!/usr/bin/env python3
"""Minimal Lumina-native SFT smoke: fine-tune on ASCR SemanticEvaluation JSON.

Avoids the heavy xllmx/fairscale dependency chain. Uses raw PyTorch training loop
with the Lumina LLaDAForMultiModalGeneration model directly.
"""
import os, sys, json, pickle, math, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from diffusers import VQModel

# Add Lumina repo
LUMINA_REPO = os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO")
sys.path.insert(0, str(Path(LUMINA_REPO).resolve()))

from model import LLaDAForMultiModalGeneration
from utils.image_utils import add_break_line, encode_img_with_breaks, calculate_vq_params, generate_crop_size_list, var_center_crop
from PIL import Image

SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,
    "eoi": 126350,
    "padding": 126339,
}

def mask_codes(codes, sch="cosine", mask=False):
    r = random.uniform(0, 1)
    if len(codes) <= 5 and not mask:
        mask_ratio = 1.0
    elif sch == "cosine":
        mask_ratio = math.cos(r * math.pi / 2)
    else:
        mask_ratio = r
    num_to_mask = max(1, int(len(codes) * mask_ratio))
    indices = random.sample(range(len(codes)), num_to_mask)
    masked = codes[:]
    labels = [-100] * len(codes)
    for idx in indices:
        labels[idx] = codes[idx]
        masked[idx] = SPECIAL_TOKENS["mask_token"]
    return masked, labels

class ASCR_SFT_Dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, vqvae, max_seq_len=4096, image_size=1024):
        self.tokenizer = tokenizer
        self.vqvae = vqvae
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def _encode_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        crop_size_list = generate_crop_size_list((self.image_size // 32) ** 2, 32)
        img = var_center_crop(img, crop_size_list=crop_size_list)
        iw, ih = img.size
        vae_scale = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        seq_len, newline_every, token_grid_h, token_grid_w = calculate_vq_params(ih, iw, vae_scale)
        tokens = encode_img_with_breaks(img, vqvae=self.vqvae)
        return tokens, token_grid_h, token_grid_w

    def __getitem__(self, idx):
        item = self.data[idx]
        # Load pre-tokenized image
        with open(item["user_image"], "rb") as f:
            img_data = pickle.load(f)
        img_tokens = img_data["input_ids"]
        h, w = img_data["height"] // 16, img_data["width"] // 16
        img_tokens = add_break_line(img_tokens, h, w, new_number=SPECIAL_TOKENS["newline_token"])

        # Build instruction
        instruction = (
            "<system>" + item["system_prompt"] + "</system>"
            + "<user>" + item["user_prompt"] + "</user>"
        )
        inst_ids = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
        inst_ids = inst_ids[:-1] + [SPECIAL_TOKENS["boi"]] + img_tokens + [SPECIAL_TOKENS["eoi"]] + inst_ids[-1:]
        inst_labels = [-100] * len(inst_ids)

        # Build answer with masking
        answer_text = item["answer_text"] + "</answer>"
        ans_ids = self.tokenizer(answer_text, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
        ans_ids, ans_labels = mask_codes(ans_ids)
        pad_len = 1024 - len(ans_ids)
        pad_ids = [SPECIAL_TOKENS["padding"]] * pad_len
        pad_ids, pad_labels = mask_codes(pad_ids, mask=True)

        all_ids = inst_ids + [SPECIAL_TOKENS["answer_start"]] + ans_ids + pad_ids
        all_labels = inst_labels + [-100] + ans_labels + pad_labels

        # Truncate/pad to max_seq_len
        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[:self.max_seq_len]
            all_labels = all_labels[:self.max_seq_len]
        else:
            pad_n = self.max_seq_len - len(all_ids)
            all_ids += [SPECIAL_TOKENS["padding"]] * pad_n
            all_labels += [-100] * pad_n

        return torch.tensor(all_ids), torch.tensor(all_labels)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    vqvae = VQModel.from_pretrained(args.checkpoint, subfolder="vqvae").to(device)
    model.train()

    # Dataset
    dataset = ASCR_SFT_Dataset(args.data_jsonl, tokenizer, vqvae, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    print(f"Training {len(dataset)} examples, {args.epochs} epochs...")
    for epoch in range(args.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Lumina expects list-of-lists: [seq of token ids]
            loss = model(input_ids=input_ids.tolist(), labels=labels.tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"  epoch {epoch} step {step}: loss={loss.item():.4f}")

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} done: avg_loss={avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/lumina-dimoo")
    parser.add_argument("--data_jsonl", default="outputs/stage2_lumina_native/sft_smoke/lumina_format/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/stage2_lumina_native/sft_smoke/checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    args = parser.parse_args()
    train(args)

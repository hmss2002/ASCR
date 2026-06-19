#!/usr/bin/env python3
"""LoRA fine-tune Lumina-DiMOO on ASCR SemanticEvaluation JSON.

Uses peft LoRA to train only ~1% of parameters, fitting in single 45GB GPU.
Lower resolution (512x512) to reduce memory further.
"""
import os, sys, json, pickle, math, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from diffusers import VQModel
from peft import LoraConfig, get_peft_model, TaskType

LUMINA_REPO = os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO")
sys.path.insert(0, str(Path(LUMINA_REPO).resolve()))

from model import LLaDAForMultiModalGeneration
from utils.image_utils import add_break_line, encode_img_with_breaks, calculate_vq_params, generate_crop_size_list, var_center_crop
from PIL import Image

SP = {
    "mask": 126336, "newline": 126084, "answer_start": 126354,
    "answer_end": 126355, "boi": 126349, "eoi": 126350, "padding": 126339,
}

def mask_codes(codes, sch="cosine", force_mask=False):
    r = random.uniform(0, 1)
    if len(codes) <= 5 and not force_mask:
        mask_ratio = 1.0
    elif sch == "cosine":
        mask_ratio = math.cos(r * math.pi / 2)
    else:
        mask_ratio = r
    n = max(1, int(len(codes) * mask_ratio))
    idxs = random.sample(range(len(codes)), n)
    masked, labels = codes[:], [-100] * len(codes)
    for i in idxs:
        labels[i], masked[i] = codes[i], SP["mask"]
    return masked, labels

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, vqvae, max_seq_len=2048, image_size=512):
        self.tokenizer = tokenizer
        self.vqvae = vqvae
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Load pre-tokenized image (from 1024px) - we'll resize at VQ level
        with open(item["user_image"], "rb") as f:
            img_data = pickle.load(f)
        img_tokens = img_data["input_ids"]
        h, w = img_data["height"] // 16, img_data["width"] // 16
        # Take center crop at token level for lower resolution
        if self.image_size < 1024:
            new_h = new_w = self.image_size // 16
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            cropped = []
            for row in range(start_h, start_h + new_h):
                cropped.extend(img_tokens[row * w + start_w : row * w + start_w + new_w])
            img_tokens = cropped
            h, w = new_h, new_w
        img_tokens = add_break_line(img_tokens, h, w, new_number=SP["newline"])

        instruction = (
            "<system>" + item["system_prompt"] + "</system>"
            + "<user>" + item["user_prompt"] + "</user>"
        )
        inst_ids = self.tokenizer(instruction, truncation=True, max_length=512, padding=False, return_tensors="pt").input_ids[0].tolist()
        inst_ids = inst_ids[:-1] + [SP["boi"]] + img_tokens + [SP["eoi"]] + inst_ids[-1:]
        inst_labels = [-100] * len(inst_ids)

        answer_text = item["answer_text"] + "</answer>"
        ans_ids = self.tokenizer(answer_text, truncation=True, max_length=512, padding=False, return_tensors="pt").input_ids[0].tolist()
        ans_ids, ans_labels = mask_codes(ans_ids)
        pad_len = 512 - len(ans_ids)
        pad_ids = [SP["padding"]] * pad_len
        pad_ids, pad_labels = mask_codes(pad_ids, force_mask=True)

        all_ids = inst_ids + [SP["answer_start"]] + ans_ids + pad_ids
        all_labels = inst_labels + [-100] + ans_labels + pad_labels

        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[:self.max_seq_len]
            all_labels = all_labels[:self.max_seq_len]
        else:
            pad_n = self.max_seq_len - len(all_ids)
            all_ids += [SP["padding"]] * pad_n
            all_labels += [-100] * pad_n

        return torch.tensor(all_ids), torch.tensor(all_labels)

def train(args):
    device = torch.device("cuda")
    print(f"Device: {device}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    vqvae = VQModel.from_pretrained(args.checkpoint, subfolder="vqvae").to(device)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SFTDataset(args.data_jsonl, tokenizer, vqvae, max_seq_len=args.max_seq_len, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f"Training {len(dataset)} examples, {args.epochs} epochs, image_size={args.image_size}...")
    for epoch in range(args.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            loss = model(input_ids=input_ids.tolist(), labels=labels.tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 5 == 0:
                print(f"  epoch {epoch} step {step}: loss={loss.item():.4f}")
        print(f"Epoch {epoch} done: avg_loss={total_loss/max(1,len(dataloader)):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/lumina-dimoo")
    parser.add_argument("--data_jsonl", default="outputs/stage2_lumina_native/sft_smoke/lumina_format/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/stage2_lumina_native/sft_smoke/lora_checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()
    train(args)

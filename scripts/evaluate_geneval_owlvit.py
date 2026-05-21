#!/usr/bin/env python3
"""
GenEval evaluator using OWLViT (HuggingFace transformers) instead of mmdet.
Produces results.jsonl compatible with external/geneval/evaluation/summary_scores.py.

Supports sharding across multiple GPUs:
    CUDA_VISIBLE_DEVICES=0 python ... --shard-id 0 --num-shards 8 --outfile results.shard0.jsonl
    ...
    CUDA_VISIBLE_DEVICES=7 python ... --shard-id 7 --num-shards 8 --outfile results.shard7.jsonl
    cat results.shard*.jsonl > results.jsonl

Usage (single GPU):
    python scripts/evaluate_geneval_owlvit.py IMAGEDIR --outfile results.jsonl \
        --model-path models/owlvit-base-patch32

Setup (run once on login node with internet):
    python scripts/download_owlvit_model.py
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-path", type=str, default="models/owlvit-base-patch32",
                        help="Local path or HF model ID for OWLViT")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Detection confidence threshold")
    parser.add_argument("--color-threshold", type=float, default=0.0,
                        help="Min logit difference for color classification (0 = argmax)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="0-indexed shard index for this worker")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (1 = no sharding)")
    return parser.parse_args()


def load_models(model_path):
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    print(f"Loading OWLViT from {model_path} ...", flush=True)
    processor = OwlViTProcessor.from_pretrained(model_path)
    model = OwlViTForObjectDetection.from_pretrained(model_path).to(DEVICE)
    model.eval()
    print(f"OWLViT loaded on {DEVICE}", flush=True)
    return model, processor


def detect_objects(model, processor, image: Image.Image, class_names: list,
                   threshold: float) -> dict:
    if not class_names:
        return {}
    texts = [[f"a photo of a {c}" for c in class_names]]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[image.height, image.width]], device=DEVICE)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        text_labels=texts,
        target_sizes=target_sizes,
        threshold=threshold,
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()

    detected: dict = {c: [] for c in class_names}
    for box, score, label in sorted(zip(boxes, scores, labels), key=lambda x: -x[1]):
        cls = class_names[int(label)]
        detected[cls].append((box.tolist(), float(score)))
    return detected


def classify_color(model, processor, image: Image.Image, box_xyxy) -> str:
    x1, y1, x2, y2 = [max(0, int(v)) for v in box_xyxy]
    crop = image.crop((x1, y1, x2, y2))
    if crop.width < 4 or crop.height < 4:
        return "unknown"
    crop = crop.resize((224, 224), Image.BICUBIC)

    color_texts = [[f"a photo of a {c} object" for c in COLORS]]
    inputs = processor(text=color_texts, images=crop, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        owlvit = model.owlvit
        text_out = owlvit.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        text_embeds = text_out.pooler_output
        text_embeds = model.owlvit.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        vision_out = owlvit.vision_model(pixel_values=inputs["pixel_values"])
        image_embeds = vision_out.pooler_output
        image_embeds = model.owlvit.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        sim = (image_embeds @ text_embeds.T)[0]
        color_idx = sim.argmax().item()
    return COLORS[color_idx]


def relative_position(obj_box, ref_box):
    ox = (obj_box[0] + obj_box[2]) / 2
    oy = (obj_box[1] + obj_box[3]) / 2
    rx = (ref_box[0] + ref_box[2]) / 2
    ry = (ref_box[1] + ref_box[3]) / 2
    dx, dy = ox - rx, oy - ry
    norm = max(1e-6, (dx ** 2 + dy ** 2) ** 0.5)
    dx, dy = dx / norm, dy / norm
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def evaluate(image, detected_map, metadata):
    correct = True
    reason = []
    matched_groups = []

    for req in metadata.get("include", []):
        classname = req["class"]
        found = detected_map.get(classname, [])[:req["count"]]
        matched = True

        if len(found) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found)}")
        else:
            if "color" in req:
                colors = [classify_color(model_ref, proc_ref, image, box) for box, _ in found]
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, "
                        f"found colors={colors}"
                    )
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if target_group >= len(matched_groups) or matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} position check")
                else:
                    for obj_box, _ in found:
                        for ref_box, _ in matched_groups[target_group]:
                            true_rels = relative_position(obj_box, ref_box)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel}, found {true_rels}"
                                )
                                break
                        if not matched:
                            break

        matched_groups.append(found if matched else None)

    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(detected_map.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(f"unexpected {classname}>={req['count']}")

    return correct, "\n".join(reason)


# Global refs set in main()
model_ref = None
proc_ref = None


def evaluate_image(filepath, metadata, threshold):
    image = ImageOps.exif_transpose(Image.open(filepath).convert("RGB"))
    class_names = list({req["class"] for req in metadata.get("include", []) + metadata.get("exclude", [])})
    if not class_names:
        return {"filename": filepath, "tag": metadata["tag"], "prompt": metadata["prompt"],
                "correct": True, "reason": "", "metadata": json.dumps(metadata), "details": "{}"}
    detected = detect_objects(model_ref, proc_ref, image, class_names, threshold)
    is_correct, reason = evaluate(image, detected, metadata)
    return {
        "filename": filepath,
        "tag": metadata["tag"],
        "prompt": metadata["prompt"],
        "correct": is_correct,
        "reason": reason,
        "metadata": json.dumps(metadata),
        "details": json.dumps({k: [[b, s] for b, s in v] for k, v in detected.items()}),
    }


def main():
    global model_ref, proc_ref
    args = parse_args()

    if args.shard_id >= args.num_shards:
        print(f"ERROR: shard-id {args.shard_id} >= num-shards {args.num_shards}", file=sys.stderr)
        sys.exit(1)

    model_ref, proc_ref = load_models(args.model_path)
    full_results = []

    subfolders = sorted(
        [f for f in os.listdir(args.imagedir)
         if os.path.isdir(os.path.join(args.imagedir, f)) and f.isdigit()],
        key=int,
    )
    # Apply sharding: each worker handles every num_shards-th subfolder
    my_subfolders = [s for i, s in enumerate(subfolders) if i % args.num_shards == args.shard_id]

    print(
        f"[shard {args.shard_id}/{args.num_shards}] "
        f"Processing {len(my_subfolders)}/{len(subfolders)} subfolders "
        f"from {args.imagedir}",
        flush=True,
    )

    for subfolder in my_subfolders:
        folderpath = os.path.join(args.imagedir, subfolder)
        meta_path = os.path.join(folderpath, "metadata.jsonl")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as fp:
            meta = json.load(fp)
        samples_dir = os.path.join(folderpath, "samples")
        if not os.path.isdir(samples_dir):
            continue
        for imagename in os.listdir(samples_dir):
            imagepath = os.path.join(samples_dir, imagename)
            if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                continue
            try:
                result = evaluate_image(imagepath, meta, args.threshold)
                full_results.append(result)
                status = "+" if result["correct"] else "-"
                print(f"  [{subfolder}/{imagename}] {status} {meta['prompt'][:60]}", flush=True)
            except Exception as exc:
                print(f"  ERROR [{subfolder}/{imagename}]: {exc}", file=sys.stderr)

    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(full_results).to_json(fp, orient="records", lines=True)

    total = len(full_results)
    correct = sum(1 for r in full_results if r["correct"])
    print(
        f"[shard {args.shard_id}/{args.num_shards}] "
        f"Done: {correct}/{total} correct ({100*correct/max(1,total):.1f}%) → {args.outfile}"
    )


if __name__ == "__main__":
    main()

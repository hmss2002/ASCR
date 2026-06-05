#!/usr/bin/env python3
"""
GenEval evaluator using OWLViT + optional DETR detector (HuggingFace transformers) instead of mmdet.
Produces results.jsonl compatible with external/geneval/evaluation/summary_scores.py.

Supports sharding across multiple GPUs:
    CUDA_VISIBLE_DEVICES=0 python ... --shard-id 0 --num-shards 8 --outfile results.shard0.jsonl
    ...
    CUDA_VISIBLE_DEVICES=7 python ... --shard-id 7 --num-shards 8 --outfile results.shard7.jsonl
    cat results.shard*.jsonl > results.jsonl

Usage (single GPU):
    python scripts/judge/evaluate_geneval_owlvit.py IMAGEDIR --outfile results.jsonl \
        --model-path models/owlvit-base-patch32

Setup (run once on login node with internet):
    python scripts/setup/download_owlvit_model.py
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
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Detection confidence threshold")
    parser.add_argument("--color-threshold", type=float, default=0.0,
                        help="Min logit difference for color classification (0 = argmax)")
    parser.add_argument("--counting-threshold", type=float, default=0.15,
                        help="Stricter detection threshold for counting tag (avoid FP inflating count)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="0-indexed shard index for this worker")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (1 = no sharding)")
    parser.add_argument("--detector", type=str, default="owlvit",
                        choices=["owlvit", "detr"],
                        help="Object detector: 'owlvit' (default) or 'detr' (COCO-trained, better for color_attr)")
    parser.add_argument("--detr-path", type=str, default="models/detr-resnet-50",
                        help="Local path or HF model ID for DETR (used when --detector detr)")
    parser.add_argument("--detr-threshold", type=float, default=0.5,
                        help="Detection threshold for DETR (default 0.5)")
    return parser.parse_args()


def load_models(model_path):
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    print(f"Loading OWLViT from {model_path} ...", flush=True)
    processor = OwlViTProcessor.from_pretrained(model_path)
    model = OwlViTForObjectDetection.from_pretrained(model_path).to(DEVICE)
    model.eval()
    print(f"OWLViT loaded on {DEVICE}", flush=True)
    return model, processor


# GenEval uses some different class names than COCO 80-class standard
GENEVAL_TO_COCO = {
    "computer keyboard": "keyboard",
    "computer mouse": "mouse",
    "tv remote": "remote",
}
COCO_TO_GENEVAL = {v: k for k, v in GENEVAL_TO_COCO.items()}


def load_detr_model(detr_path):
    from transformers import DetrImageProcessor, DetrForObjectDetection
    print(f"Loading DETR from {detr_path} ...", flush=True)
    processor = DetrImageProcessor.from_pretrained(detr_path)
    model = DetrForObjectDetection.from_pretrained(detr_path).to(DEVICE)
    model.eval()
    print(f"DETR loaded on {DEVICE}", flush=True)
    return model, processor



def _nms_numpy(boxes_xyxy, scores, iou_thr=0.5):
    """Per-class NMS in numpy. boxes: (N,4) xyxy, scores: (N,)."""
    if len(boxes_xyxy) == 0:
        return []
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def _apply_nms_per_class(detected: dict, iou_thr: float = 0.5) -> dict:
    """Apply per-class NMS to a {class: [(box, score), ...]} dict."""
    out = {}
    for cls, items in detected.items():
        if not items:
            out[cls] = []
            continue
        boxes = [b for b, _ in items]
        scores = [s for _, s in items]
        keep = _nms_numpy(boxes, scores, iou_thr=iou_thr)
        out[cls] = [items[k] for k in keep]
    return out


def detect_objects_detr(detr_model, detr_proc, image: Image.Image, class_names: list,
                        threshold: float) -> dict:
    """Detect objects using DETR (COCO-trained, color-agnostic)."""
    if not class_names:
        return {}
    inputs = detr_proc(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = detr_model(**inputs)
    target_sizes = torch.tensor([[image.height, image.width]], device=DEVICE)
    results = detr_proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold)[0]

    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    boxes = results["boxes"].cpu().numpy()

    detected: dict = {c: [] for c in class_names}
    for score, label, box in sorted(zip(scores, labels, boxes), key=lambda x: -x[0]):
        coco_name = detr_model.config.id2label[int(label)]
        # Map COCO name → GenEval name (e.g. "keyboard" → "computer keyboard")
        geneval_name = COCO_TO_GENEVAL.get(coco_name, coco_name)
        if geneval_name in detected:
            detected[geneval_name].append((box.tolist(), float(score)))
    return _apply_nms_per_class(detected, iou_thr=0.5)


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
    return _apply_nms_per_class(detected, iou_thr=0.5)


def classify_color(model, processor, image, box_xyxy) -> str:
    """Classify dominant color in a bounding-box region via HSV pixel statistics.

    OWLViT / CLIP pooler embeddings cannot reliably discriminate colors
    (they produce near-identical similarities, always returning a single bias
    color regardless of input). This pixel-based approach mirrors the
    classical CV method used by many color benchmarks.

    Args ``model`` and ``processor`` are accepted for backward compatibility
    but unused.
    """
    x1, y1, x2, y2 = [max(0, int(v)) for v in box_xyxy]
    crop = image.crop((x1, y1, x2, y2))
    if crop.width < 4 or crop.height < 4:
        return "unknown"
    crop = crop.convert("RGB").resize((64, 64), Image.BILINEAR)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    delta = mx - mn
    h = np.zeros_like(mx)
    m = delta > 1e-6
    rm = (mx == r) & m
    gm = (mx == g) & m & ~rm
    bm = (mx == b) & m & ~rm & ~gm
    # avoid divide-by-zero where delta==0
    with np.errstate(divide="ignore", invalid="ignore"):
        h[rm] = (60.0 * ((g[rm] - b[rm]) / delta[rm])) % 360.0
        h[gm] = (60.0 * ((b[gm] - r[gm]) / delta[gm]) + 120.0)
        h[bm] = (60.0 * ((r[bm] - g[bm]) / delta[bm]) + 240.0)

    flat_h = h.flatten()
    flat_s = s.flatten()
    flat_v = v.flatten()

    # Vectorized achromatic / hue-bin classification
    labels = np.full(flat_h.shape, None, dtype=object)
    # Black: very low value
    is_black = flat_v < 0.18
    # Achromatic (low saturation): white if bright, black if dark, else skip
    achrom = (flat_s < 0.12) & ~is_black
    is_white = achrom & (flat_v > 0.75)
    is_black2 = achrom & (flat_v < 0.35)
    achrom_skip = achrom & ~is_white & ~is_black2
    labels[is_black] = "black"
    labels[is_white] = "white"
    labels[is_black2] = "black"
    # Chromatic = non-black, non-achromatic
    chrom = ~(is_black | achrom)
    # Brown: low-V orange-yellow hue range
    is_brown = chrom & (flat_h >= 10) & (flat_h < 50) & (flat_v < 0.55) & (flat_s > 0.25)
    # Pink: high-V, mid-S magenta/red range
    is_pink = chrom & ((flat_h < 15) | (flat_h >= 320)) & (flat_s < 0.55) & (flat_v > 0.7) & ~is_brown
    remaining = chrom & ~is_brown & ~is_pink
    labels[is_brown] = "brown"
    labels[is_pink] = "pink"
    # Hue-bin remaining chromatic pixels
    hh = flat_h
    labels[remaining & ((hh < 15) | (hh >= 345))] = "red"
    labels[remaining & (hh >= 15) & (hh < 40)] = "orange"
    labels[remaining & (hh >= 40) & (hh < 70)] = "yellow"
    labels[remaining & (hh >= 70) & (hh < 165)] = "green"
    labels[remaining & (hh >= 165) & (hh < 250)] = "blue"
    labels[remaining & (hh >= 250) & (hh < 290)] = "purple"
    labels[remaining & (hh >= 290) & (hh < 345)] = "pink"

    valid = labels[labels != None]
    if valid.size == 0:
        return "unknown"
    # Most common label
    vals, counts = np.unique(valid, return_counts=True)
    return str(vals[counts.argmax()])

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
model_ref = None    # OWLViT model (for classify_color)
proc_ref = None     # OWLViT processor
det_model_ref = None  # Detector model (None → use OWLViT; set → use DETR)
det_proc_ref = None   # Detector processor


def evaluate_image(filepath, metadata, threshold, counting_threshold=0.15):
    image = ImageOps.exif_transpose(Image.open(filepath).convert("RGB"))
    class_names = list({req["class"] for req in metadata.get("include", []) + metadata.get("exclude", [])})
    if not class_names:
        return {"filename": filepath, "tag": metadata["tag"], "prompt": metadata["prompt"],
                "correct": True, "reason": "", "metadata": json.dumps(metadata), "details": "{}"}
    # counting suffers from low-threshold false positives (exclude rule fails);
    # use a stricter threshold for counting tag.
    thr = counting_threshold if metadata.get("tag") == "counting" else threshold
    if det_model_ref is not None:
        detected = detect_objects_detr(det_model_ref, det_proc_ref, image, class_names, thr)
    else:
        detected = detect_objects(model_ref, proc_ref, image, class_names, thr)
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
    global model_ref, proc_ref, det_model_ref, det_proc_ref
    args = parse_args()

    if args.shard_id >= args.num_shards:
        print(f"ERROR: shard-id {args.shard_id} >= num-shards {args.num_shards}", file=sys.stderr)
        sys.exit(1)

    model_ref, proc_ref = load_models(args.model_path)
    if args.detector == "detr":
        det_model_ref, det_proc_ref = load_detr_model(args.detr_path)
        threshold = args.detr_threshold
        print(f"Using DETR detector (threshold={threshold})", flush=True)
    else:
        threshold = args.threshold
        print(f"Using OWLViT detector (threshold={threshold})", flush=True)
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
                result = evaluate_image(imagepath, meta, threshold, counting_threshold=args.counting_threshold)
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

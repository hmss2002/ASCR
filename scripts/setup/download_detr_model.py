#!/usr/bin/env python3
"""
Download facebook/detr-resnet-50 to models/detr-resnet-50 for offline use.
Run once on a login node with internet access.
"""
import os
from transformers import DetrImageProcessor, DetrForObjectDetection

MODEL_ID = "facebook/detr-resnet-50"
SAVE_DIR = "models/detr-resnet-50"

print(f"Downloading {MODEL_ID} → {SAVE_DIR}")
processor = DetrImageProcessor.from_pretrained(MODEL_ID)
model = DetrForObjectDetection.from_pretrained(MODEL_ID)
os.makedirs(SAVE_DIR, exist_ok=True)
processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)
print(f"Done. Saved to {SAVE_DIR}")

import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path


SELECTOR_CHECKPOINT_SCHEMA = "stage2.selector_checkpoint.v1"


def _sample_key(sample):
    prompt = str(sample.get("prompt", "")).strip()
    iteration = int(sample.get("iteration", 0))
    decoded_image = str(sample.get("decoded_image", "")).strip()
    return f"{prompt}||{iteration}||{decoded_image}"


def normalize_prompt_tokens(prompt):
    return re.findall(r"[a-z0-9]+", str(prompt or "").lower())


def prompt_hash_features(prompt, dim=256):
    values = [0.0] * int(dim)
    tokens = normalize_prompt_tokens(prompt)
    if not tokens:
        return values
    for token in tokens:
        index = zlib.crc32(token.encode("utf-8")) % int(dim)
        values[index] += 1.0
    scale = float(sum(values)) or 1.0
    return [value / scale for value in values]


def cell_labels_to_multi_hot(labels, grid_size=4):
    values = [0.0] * (int(grid_size) * int(grid_size))
    for label in labels or []:
        text = str(label).strip().upper()
        if len(text) < 2:
            continue
        row = ord(text[0]) - ord("A")
        try:
            col = int(text[1:]) - 1
        except ValueError:
            continue
        if 0 <= row < int(grid_size) and 0 <= col < int(grid_size):
            values[row * int(grid_size) + col] = 1.0
    return values


def multi_hot_to_cell_labels(values, grid_size=4, threshold=0.5, max_selected_cells=8):
    scored = []
    for index, score in enumerate(values):
        score = float(score)
        if score < float(threshold):
            continue
        row = index // int(grid_size)
        col = index % int(grid_size)
        scored.append((score, f"{chr(ord('A') + row)}{col + 1}"))
    scored.sort(reverse=True)
    return [label for _, label in scored[:int(max_selected_cells)]]


def load_image_tensor(image_path, image_size=64):
    try:
        import numpy as np
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Learned selector image loading requires numpy, torch, and Pillow") from exc
    with Image.open(image_path).convert("RGB") as image:
        resized = image.resize((int(image_size), int(image_size)))
        array = np.asarray(resized, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


@dataclass
class SemanticReopeningSelectorModel:
    model_type: str
    metadata: dict

    def to_dict(self):
        return {
            "schema_version": SELECTOR_CHECKPOINT_SCHEMA,
            "model_type": self.model_type,
            "metadata": self.metadata,
        }

    def save(self, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


class TeacherReplaySelectorModel(SemanticReopeningSelectorModel):
    def __init__(self, metadata=None):
        super().__init__("teacher_replay", metadata or {})

    def predict_from_teacher_trace(self, sample):
        return sample.get("projected_token_mask")


class DatasetReplaySelectorModel(SemanticReopeningSelectorModel):
    def __init__(self, replay_index=None, metadata=None):
        super().__init__("dataset_replay", metadata or {})
        self.replay_index = replay_index or {}

    @classmethod
    def from_samples(cls, samples, metadata=None):
        replay_index = {}
        for sample in samples:
            replay_index[_sample_key(sample)] = {
                "selected_4x4_cells": sample.get("selected_4x4_cells", []),
                "projected_token_mask": sample.get("projected_token_mask"),
                "correction_instruction": sample.get("correction_instruction"),
            }
        return cls(replay_index=replay_index, metadata=metadata or {})

    def lookup(self, sample):
        return self.replay_index.get(_sample_key(sample))

    def to_dict(self):
        payload = super().to_dict()
        payload["replay_index_size"] = len(self.replay_index)
        return payload

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = super().save(output_dir / "selector_checkpoint.json")
        replay_path = output_dir / "replay_index.jsonl"
        with replay_path.open("w", encoding="utf-8") as handle:
            for key, payload in sorted(self.replay_index.items()):
                json.dump({"sample_key": key, **payload}, handle, sort_keys=True)
                handle.write("\n")
        return {"checkpoint_path": str(checkpoint_path), "replay_index_path": str(replay_path)}


def build_learned_selector_network(prompt_hash_dim=256, image_size=64, hidden_dim=128, grid_size=4):
    try:
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("Learned selector network construction requires torch") from exc
    conv_out = max(4, int(image_size) // 8)
    conv_dim = 32 * conv_out * conv_out
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(conv_dim + int(prompt_hash_dim) + 1, int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(grid_size) * int(grid_size) + 1),
    )


class LearnedCoarseSelectorModel(SemanticReopeningSelectorModel):
    def __init__(self, prompt_hash_dim=256, image_size=64, hidden_dim=128, grid_size=4, max_selected_cells=8, metadata=None):
        super().__init__("learned_coarse_selector", metadata or {})
        self.prompt_hash_dim = int(prompt_hash_dim)
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)
        self.max_selected_cells = int(max_selected_cells)
        self._network = None

    def network(self):
        if self._network is None:
            self._network = build_learned_selector_network(
                prompt_hash_dim=self.prompt_hash_dim,
                image_size=self.image_size,
                hidden_dim=self.hidden_dim,
                grid_size=self.grid_size,
            )
        return self._network

    def checkpoint_payload(self, state_dict=None, metrics=None):
        return {
            "model_type": self.model_type,
            "config": {
                "prompt_hash_dim": self.prompt_hash_dim,
                "image_size": self.image_size,
                "hidden_dim": self.hidden_dim,
                "grid_size": self.grid_size,
                "max_selected_cells": self.max_selected_cells,
            },
            "metadata": self.metadata,
            "metrics": metrics or {},
            "state_dict": state_dict,
        }

    def save(self, output_dir, metrics=None):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Saving a learned selector checkpoint requires torch") from exc
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "learned_selector.pt"
        torch.save(self.checkpoint_payload(state_dict=self.network().state_dict(), metrics=metrics), str(weights_path))
        checkpoint = {
            **super().to_dict(),
            "weights_path": str(weights_path),
            "config": {
                "prompt_hash_dim": self.prompt_hash_dim,
                "image_size": self.image_size,
                "hidden_dim": self.hidden_dim,
                "grid_size": self.grid_size,
                "max_selected_cells": self.max_selected_cells,
            },
            "metrics": metrics or {},
        }
        checkpoint_path = output_dir / "selector_checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
        return {"checkpoint_path": str(checkpoint_path), "weights_path": str(weights_path)}

    @classmethod
    def load(cls, checkpoint_path, map_location="cpu"):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Loading a learned selector checkpoint requires torch") from exc
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "selector_checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        config = checkpoint.get("config", {})
        model = cls(
            prompt_hash_dim=int(config.get("prompt_hash_dim", 256)),
            image_size=int(config.get("image_size", 64)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            grid_size=int(config.get("grid_size", 4)),
            max_selected_cells=int(config.get("max_selected_cells", 8)),
            metadata=checkpoint.get("metadata", {}),
        )
        weights_path = checkpoint.get("weights_path")
        if not weights_path:
            raise ValueError(f"Learned selector checkpoint is missing weights_path: {checkpoint_path}")
        payload = torch.load(weights_path, map_location=map_location)
        model.network().load_state_dict(payload["state_dict"])
        model.network().eval()
        return model

    def predict(self, prompt, grid_image_path, iteration=0, device="cpu", error_threshold=0.5, cell_threshold=0.5):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Learned selector inference requires torch") from exc
        self.network().to(device)
        self.network().eval()
        prompt_tensor = torch.tensor([prompt_hash_features(prompt, self.prompt_hash_dim)], dtype=torch.float32, device=device)
        image_tensor = load_image_tensor(grid_image_path, self.image_size).unsqueeze(0).to(device)
        iteration_tensor = torch.tensor([[float(iteration)]], dtype=torch.float32, device=device)
        features = torch.cat([self.network()[:7](image_tensor), prompt_tensor, iteration_tensor], dim=1)
        logits = self.network()[7:](features)
        cell_logits = logits[:, :-1]
        error_logit = logits[:, -1]
        cell_probs = torch.sigmoid(cell_logits)[0].detach().cpu().tolist()
        error_probability = float(torch.sigmoid(error_logit)[0].detach().cpu().item())
        labels = multi_hot_to_cell_labels(
            cell_probs,
            grid_size=self.grid_size,
            threshold=cell_threshold,
            max_selected_cells=self.max_selected_cells,
        )
        if error_probability >= float(error_threshold) and not labels:
            top_index = max(range(len(cell_probs)), key=lambda index: cell_probs[index])
            row = top_index // self.grid_size
            col = top_index % self.grid_size
            labels = [f"{chr(ord('A') + row)}{col + 1}"]
        return {
            "has_error": error_probability >= float(error_threshold) and bool(labels),
            "error_probability": error_probability,
            "cell_probabilities": cell_probs,
            "selected_cells": labels,
        }

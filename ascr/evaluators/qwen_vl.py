import json
from pathlib import Path

from ascr.core.schemas import SemanticEvaluation, parse_semantic_evaluation
from ascr.evaluators.base import SemanticEvaluator


def _extract_json_object(text):
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Qwen-VL response did not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == chr(34):
                in_string = False
            continue
        if char == chr(34):
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:index + 1])
    raise ValueError("Qwen-VL response JSON object was not closed")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _short_text(value, default="semantic mismatch"):
    if value is None:
        return default
    return " ".join(str(value).splitlines()).strip() or default


def _normalize_region(region, default_reason="semantic mismatch", default_error_type="semantic"):
    if not isinstance(region, dict):
        return region
    normalized = dict(region)
    if "cells" not in normalized:
        for key in ("grid_cells", "selected_cells", "cell_labels", "cell", "label"):
            if key in normalized:
                normalized["cells"] = _as_list(normalized[key])
                break
    normalized.setdefault("reason", _short_text(normalized.get("reason", normalized.get("issue", default_reason))))
    normalized.setdefault("confidence", float(normalized.get("score", 1.0)))
    normalized.setdefault("error_type", str(normalized.get("error_type", normalized.get("type", default_error_type))))
    normalized.setdefault("action", "reopen")
    return normalized


def _normalize_payload(payload, max_selected_cells=6):
    if not isinstance(payload, dict):
        raise ValueError("Qwen-VL JSON payload must be an object")
    normalized = dict(payload)
    if "has_error" not in normalized:
        if "match" in normalized:
            normalized["has_error"] = not bool(normalized.get("match"))
        elif "is_match" in normalized:
            normalized["has_error"] = not bool(normalized.get("is_match"))
        elif "error_present" in normalized:
            normalized["has_error"] = bool(normalized.get("error_present"))
    normalized.setdefault("summary", _short_text(normalized.get("summary", normalized.get("diagnosis", "Qwen-VL semantic judgment"))))
    regions = normalized.get("regions", normalized.get("selected_regions"))
    if regions is None:
        raw_cells = normalized.get("grid_cells", normalized.get("cells", normalized.get("selected_cells")))
        if raw_cells:
            regions = [{"cells": _as_list(raw_cells), "reason": normalized["summary"], "confidence": normalized.get("confidence", 1.0), "error_type": normalized.get("error_type", "semantic"), "action": "reopen"}]
    if regions is None:
        candidates = []
        for error in _as_list(normalized.get("errors")):
            if not isinstance(error, dict):
                continue
            cells = error.get("cells", error.get("grid_cells", error.get("selected_cells")))
            if cells:
                candidates.append({"cells": _as_list(cells), "reason": _short_text(error.get("reason", error.get("issue", normalized["summary"]))), "confidence": error.get("confidence", 1.0), "error_type": error.get("type", error.get("error_type", "semantic")), "action": "reopen"})
        if candidates:
            regions = candidates
    if regions is not None:
        normalized["regions"] = [_normalize_region(region, normalized["summary"], normalized.get("error_type", "semantic")) for region in _as_list(regions)]
    if "correction_instruction" not in normalized:
        normalized["correction_instruction"] = str(normalized.get("suggested_fix", "Regenerate the selected grid cells so the image satisfies the original prompt while preserving correct content."))
    normalized["max_selected_cells"] = int(max_selected_cells)
    return normalized


class QwenVLEvaluator(SemanticEvaluator):
    def __init__(self, model_path="Qwen/Qwen3.6-35B-A3B", device="cuda", device_map="auto", torch_dtype="bfloat16", trust_remote_code=True, local_files_only=False, strict_json=True, grid_size=4, image_size=512, max_new_tokens=768, max_selected_cells=6, temperature=0.0, top_p=1.0, attn_implementation=None, processor_use_fast=False):
        self.model_path = model_path
        self.device = device
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        self.strict_json = bool(strict_json)
        self.grid_size = int(grid_size)
        self.image_size = int(image_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_selected_cells = int(max_selected_cells)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.attn_implementation = attn_implementation
        self.processor_use_fast = bool(processor_use_fast)
        self._processor = None
        self._model = None
        self._torch = None

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f"Missing image for Qwen-VL evaluation: {grid_image_path}")
        raw_text = ""
        try:
            raw_text = self._generate_text(self._build_question(original_prompt), grid_image_path)
            payload = _normalize_payload(_extract_json_object(raw_text), max_selected_cells=self.max_selected_cells)
            evaluation = parse_semantic_evaluation(payload, grid_size=self.grid_size, max_selected_cells=self.max_selected_cells)
        except Exception as exc:
            return SemanticEvaluation.abstain(f"Qwen-VL evaluator failed: {exc}", raw={"qwen_vl_text": raw_text, "model_path": self.model_path})
        return SemanticEvaluation(
            has_error=evaluation.has_error,
            summary=evaluation.summary,
            regions=evaluation.regions,
            correction_instruction=evaluation.correction_instruction,
            should_abstain=evaluation.should_abstain,
            parser_error=evaluation.parser_error,
            raw={"qwen_vl_text": raw_text, "qwen_vl_payload": payload, "model_path": self.model_path},
        )

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as AutoModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModel
        except Exception as exc:
            raise RuntimeError("Qwen-VL evaluator needs torch, transformers with image-text model support, and a compatible processor") from exc
        self._torch = torch
        processor_kwargs = {"trust_remote_code": self.trust_remote_code, "local_files_only": self.local_files_only, "use_fast": self.processor_use_fast}
        model_kwargs = {"trust_remote_code": self.trust_remote_code, "local_files_only": self.local_files_only}
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if self.torch_dtype:
            if self.torch_dtype == "auto":
                model_kwargs["torch_dtype"] = "auto"
            else:
                model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self._processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        self._model = AutoModel.from_pretrained(self.model_path, **model_kwargs)
        if not self.device_map and self.device:
            self._model.to(self.device)
        self._model.eval()

    def _first_device(self):
        try:
            return next(self._model.parameters()).device
        except Exception:
            return self.device

    def _generate_text(self, question, image_path):
        self._load()
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        try:
            inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        except Exception:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[image], return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._first_device())
        generation_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        with self._torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generation_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        trimmed = generated_ids[:, input_length:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _build_question(self, original_prompt):
        labels = ", ".join(chr(65 + row) + str(col + 1) for row in range(self.grid_size) for col in range(self.grid_size))
        return " ".join([
            "You are the strict semantic evaluator for ASCR Stage 1.",
            "The image contains visible grid labels. Treat grid lines and labels as evaluation aids, not as part of the generated scene.",
            f"Original text-to-image prompt: {original_prompt}",
            f"Grid cells are {labels}. Rows A-D go top to bottom and columns 1-4 go left to right.",
            "Decide whether the image materially violates the prompt. Check objects, counts, colors, attributes, text, and spatial relations.",
            "If there is no material semantic error, return has_error false and an empty regions array.",
            f"If there is an error, choose at most {self.max_selected_cells} smallest grid cells that cover the wrong, missing, or extra content.",
            "Return only one JSON object with this schema:",
            "Required JSON keys: has_error, summary, regions, correction_instruction. Each region must include cells, reason, confidence, error_type, and action reopen.",
        ])

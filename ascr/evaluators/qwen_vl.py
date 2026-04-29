import json
from pathlib import Path

from ascr.core.schemas import SemanticEvaluation, parse_semantic_evaluation
from ascr.evaluators.base import SemanticEvaluator


_QWEN35_MOE_MODEL_TYPE = "qwen3_5_moe"
_QWEN35_MOE_NATIVE_CONFIG = "Qwen3_5MoeConfig"
_QWEN35_MOE_NATIVE_MODEL = "Qwen3_5MoeForConditionalGeneration"


def _qwen35_moe_native_error(detail=None):
    message = (
        "Qwen3.6 qwen3_5_moe checkpoints require native Transformers support. "
        "Use the .venv-qwen36 environment with Python 3.11, torch>=2.4, matching torchvision, "
        "and Transformers from the official Qwen3.5 support commit fc9137225 or newer."
    )
    if detail:
        return f"{message} Detail: {detail}"
    return message


def _is_qwen35_moe_config(config):
    return getattr(config, "model_type", None) == _QWEN35_MOE_MODEL_TYPE


def _require_native_qwen35_moe_support(config, auto_model_class):
    if not _is_qwen35_moe_config(config):
        return
    config_name = type(config).__name__
    if config_name != _QWEN35_MOE_NATIVE_CONFIG:
        raise RuntimeError(_qwen35_moe_native_error(f"loaded config class {config_name}, expected {_QWEN35_MOE_NATIVE_CONFIG}"))
    try:
        model_class = auto_model_class._model_mapping[type(config)]
    except Exception as exc:
        raise RuntimeError(_qwen35_moe_native_error("AutoModelForImageTextToText does not map qwen3_5_moe to the native model class")) from exc
    model_name = getattr(model_class, "__name__", "")
    if model_name != _QWEN35_MOE_NATIVE_MODEL:
        raise RuntimeError(_qwen35_moe_native_error(f"AutoModel maps qwen3_5_moe to {model_name}, expected {_QWEN35_MOE_NATIVE_MODEL}"))
    try:
        from transformers.utils import is_torchvision_available
    except Exception as exc:
        raise RuntimeError(_qwen35_moe_native_error("could not verify torchvision availability for the Qwen3VL processor")) from exc
    if not is_torchvision_available():
        raise RuntimeError(_qwen35_moe_native_error("torchvision is required for the Qwen3VL video processor used by Qwen3.6"))


def _final_answer_text(text):
    cleaned = str(text or "").strip()
    lowered = cleaned.lower()
    closing_positions = []
    for tag in ("</think>", "</thinking>"):
        position = lowered.rfind(tag)
        if position >= 0:
            closing_positions.append((position + len(tag), tag))
    if closing_positions:
        start, _ = max(closing_positions)
        return cleaned[start:].strip(), True
    for marker in ("FINAL_JSON:", "Final JSON:", "Final answer:", "Answer:", "JSON:"):
        position = lowered.rfind(marker.lower())
        if position >= 0:
            return cleaned[position + len(marker):].strip(), False
    return cleaned, False

def _extract_json_object(text):
    cleaned, _ = _final_answer_text(text)
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
    last_error = None
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            current = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == chr(34):
                    in_string = False
                continue
            if current == chr(34):
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:index + 1])
                    except json.JSONDecodeError as exc:
                        last_error = exc
                        break
        else:
            last_error = ValueError("Qwen-VL response JSON object was not closed")
    if last_error is not None:
        raise ValueError(f"Qwen-VL response did not contain a parseable JSON object: {last_error}")
    raise ValueError("Qwen-VL response did not contain a JSON object")

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


def _budget_regions(regions, max_selected_cells):
    normalized_regions = [region for region in regions if isinstance(region, dict)]
    budget = int(max_selected_cells)
    if budget <= 0:
        return []
    selected = [[] for _ in normalized_regions]
    seen = set()
    selected_count = 0
    while selected_count < budget:
        changed = False
        for index, region in enumerate(normalized_regions):
            for cell in _as_list(region.get("cells")):
                try:
                    key = json.dumps(cell, sort_keys=True)
                except TypeError:
                    key = str(cell).upper()
                if key in seen:
                    continue
                selected[index].append(cell)
                seen.add(key)
                selected_count += 1
                changed = True
                break
            if selected_count >= budget:
                break
        if not changed:
            break
    budgeted = []
    for region, cells in zip(normalized_regions, selected):
        if not cells:
            continue
        budgeted_region = dict(region)
        budgeted_region["cells"] = cells
        budgeted.append(budgeted_region)
    return budgeted


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
        normalized_regions = [_normalize_region(region, normalized["summary"], normalized.get("error_type", "semantic")) for region in _as_list(regions)]
        normalized["regions"] = _budget_regions(normalized_regions, max_selected_cells)
    if "correction_instruction" not in normalized:
        normalized["correction_instruction"] = str(normalized.get("suggested_fix", "Regenerate the selected grid cells so the image satisfies the original prompt while preserving correct content."))
    normalized["max_selected_cells"] = int(max_selected_cells)
    return normalized


class QwenVLEvaluator(SemanticEvaluator):
    def __init__(self, model_path="Qwen/Qwen3.6-35B-A3B", device="cuda", device_map="auto", torch_dtype="bfloat16", trust_remote_code=True, local_files_only=False, strict_json=True, grid_size=4, image_size=512, max_new_tokens=768, repair_max_new_tokens=None, max_selected_cells=6, temperature=0.0, top_p=1.0, attn_implementation=None, processor_use_fast=False, enable_thinking=True):
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
        self.repair_max_new_tokens = int(repair_max_new_tokens) if repair_max_new_tokens is not None else max(self.max_new_tokens, 384)
        self.max_selected_cells = int(max_selected_cells)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.attn_implementation = attn_implementation
        self.processor_use_fast = bool(processor_use_fast)
        self.enable_thinking = enable_thinking
        self._processor = None
        self._model = None
        self._torch = None

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f"Missing image for Qwen-VL evaluation: {grid_image_path}")
        raw_text = ""
        json_text = ""
        try:
            raw_text = self._generate_text(self._build_question(original_prompt), grid_image_path, enable_thinking=self.enable_thinking)
            json_text = raw_text
            try:
                raw_payload = _extract_json_object(json_text)
            except ValueError:
                if not self.strict_json:
                    raise
                json_text = self._generate_text(self._build_json_repair_question(original_prompt, raw_text), grid_image_path, enable_thinking=False, max_new_tokens=self.repair_max_new_tokens)
                raw_payload = _extract_json_object(json_text)
            payload = _normalize_payload(raw_payload, max_selected_cells=self.max_selected_cells)
            evaluation = parse_semantic_evaluation(payload, grid_size=self.grid_size, max_selected_cells=self.max_selected_cells)
        except Exception as exc:
            raw = {"qwen_vl_text": raw_text, "model_path": self.model_path}
            if json_text and json_text != raw_text:
                raw["qwen_vl_json_text"] = json_text
            return SemanticEvaluation.abstain(f"Qwen-VL evaluator failed: {exc}", raw=raw)
        raw = {"qwen_vl_text": raw_text, "qwen_vl_payload": payload, "model_path": self.model_path}
        if json_text != raw_text:
            raw["qwen_vl_json_text"] = json_text
        return SemanticEvaluation(
            has_error=evaluation.has_error,
            summary=evaluation.summary,
            regions=evaluation.regions,
            correction_instruction=evaluation.correction_instruction,
            should_abstain=evaluation.should_abstain,
            parser_error=evaluation.parser_error,
            raw=raw,
        )

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as AutoModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModel
        except Exception as exc:
            raise RuntimeError("Qwen-VL evaluator needs torch, transformers with image-text model support, and a compatible processor") from exc
        self._torch = torch
        processor_kwargs = {"trust_remote_code": self.trust_remote_code, "local_files_only": self.local_files_only, "use_fast": self.processor_use_fast}
        model_kwargs = {"trust_remote_code": self.trust_remote_code, "local_files_only": self.local_files_only}
        try:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code, local_files_only=self.local_files_only)
        except Exception as exc:
            raise RuntimeError(_qwen35_moe_native_error(f"AutoConfig could not load {self.model_path}: {exc}")) from exc
        _require_native_qwen35_moe_support(config, AutoModel)
        model_kwargs["config"] = config
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if self.torch_dtype:
            if self.torch_dtype == "auto":
                model_kwargs["torch_dtype"] = "auto"
            else:
                model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        except Exception as exc:
            if _is_qwen35_moe_config(config):
                raise RuntimeError(_qwen35_moe_native_error(f"AutoProcessor failed for {self.model_path}: {exc}")) from exc
            raise
        try:
            self._model = AutoModel.from_pretrained(self.model_path, **model_kwargs)
        except Exception as exc:
            if _is_qwen35_moe_config(config):
                raise RuntimeError(_qwen35_moe_native_error(f"AutoModel failed for {self.model_path}: {exc}")) from exc
            raise
        if not self.device_map and self.device:
            self._model.to(self.device)
        self._model.eval()

    def _first_device(self):
        try:
            return next(self._model.parameters()).device
        except Exception:
            return self.device

    def _apply_chat_template(self, messages, enable_thinking=None, **kwargs):
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if thinking is None:
            return self._processor.apply_chat_template(messages, **kwargs)
        try:
            return self._processor.apply_chat_template(messages, enable_thinking=bool(thinking), **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            return self._processor.apply_chat_template(messages, **kwargs)

    def _generate_text(self, question, image_path, enable_thinking=None, max_new_tokens=None):
        self._load()
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        try:
            inputs = self._apply_chat_template(messages, enable_thinking=enable_thinking, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        except Exception:
            text = self._apply_chat_template(messages, enable_thinking=enable_thinking, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[image], return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._first_device())
        generation_kwargs = {"max_new_tokens": int(max_new_tokens or self.max_new_tokens), "do_sample": self.temperature > 0}
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

    def _build_json_repair_question(self, original_prompt, previous_text):
        previous_text = " ".join(str(previous_text).split())[:1800]
        return " ".join([
            "/no_think",
            "Return one compact valid JSON object for this semantic evaluation.",
            "Your entire response must be only the JSON object and must end with }.",
            f"Original text-to-image prompt: {original_prompt}",
            "Do not add prose, markdown, code fences, or analysis.",
            "Use this schema: {\"has_error\": boolean, \"summary\": string, \"regions\": array, \"correction_instruction\": string}.",
            "Keep summary under 25 words and each reason under 10 words.",
            "If the image satisfies the prompt, return has_error false and regions [].",
            f"If there is an error, select at most {self.max_selected_cells} total grid cells across all regions; each region needs cells, reason, confidence, error_type, and action set to reopen.",
            "If the previous text is incomplete, finish the judgment instead of copying it.",
            "Previous incomplete evaluation text:",
            previous_text,
        ])

    def _build_question(self, original_prompt):
        labels = ", ".join(chr(65 + row) + str(col + 1) for row in range(self.grid_size) for col in range(self.grid_size))
        parts = [
            "You are the strict semantic evaluator for ASCR Stage 1.",
            "The image contains visible grid labels. Treat grid lines and labels as evaluation aids, not as part of the generated scene.",
            f"Original text-to-image prompt: {original_prompt}",
            f"Grid cells are {labels}. Rows A-D go top to bottom and columns 1-4 go left to right.",
            "Decide whether the image materially violates the prompt. Check objects, counts, colors, attributes, text, and spatial relations.",
            "Return exactly one valid JSON object. Start with { and end with }.",
            "Do not include markdown, code fences, or bullet points.",
            'Use this schema: {"has_error": boolean, "summary": string, "regions": array, "correction_instruction": string}.',
            "If there is no material semantic error, set has_error to false and regions to an empty array.",
            f"If there is an error, choose at most {self.max_selected_cells} smallest grid cells that cover the wrong, missing, or extra content.",
            "Each region must include cells, reason, confidence, error_type, and action set to reopen.",
        ]
        if self.enable_thinking:
            parts.insert(6, "The assistant response starts inside a <think> block; keep that block under 80 words, then close it with </think>.")
            parts.insert(7, "After </think>, write FINAL_JSON: followed immediately by exactly one valid JSON object.")
            parts.insert(8, "Do not write numbered or bulleted analysis.")
            parts.insert(9, "Stop immediately after the JSON object closing }.")
        else:
            parts.insert(0, "/no_think")
            parts.insert(6, "Do not include prose or analysis.")
        return " ".join(parts)

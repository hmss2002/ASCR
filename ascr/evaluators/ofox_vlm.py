import base64
import json
import mimetypes
import os
import time
from pathlib import Path

from ascr.core.schemas import SemanticEvaluation, parse_semantic_evaluation
from ascr.evaluators.base import SemanticEvaluator
from ascr.evaluators.qwen_vl import _extract_json_object, _normalize_payload


DEFAULT_OFOX_BASE_URL = "https://api.ofox.ai/v1"


def _normalize_base_url(value):
    text = str(value or "").strip()
    if not text:
        return DEFAULT_OFOX_BASE_URL
    normalized = text.rstrip("/")
    if normalized in {"https://ofox.ai", "https://ofox.ai/zh"}:
        return DEFAULT_OFOX_BASE_URL
    return normalized


def _as_serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _as_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_serializable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _as_serializable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _as_serializable(value.dict())
        except Exception:
            pass
    return str(value)


def _response_text(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("Ofox API returned no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("Ofox API returned empty response content")


def _image_to_data_url(image_path):
    path = Path(image_path)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _usage_from_response(response):
    return _as_serializable(getattr(response, "usage", None))


class OfoxVLMEvaluator(SemanticEvaluator):
    def __init__(
        self,
        model="bailian/qwen3.7-plus",
        base_url=None,
        api_key=None,
        base_url_env="OFOX_BASE_URL",
        api_key_env="OFOX_API_KEY",
        strict_json=True,
        grid_size=4,
        image_size=1024,
        max_input_tokens=32768,
        max_new_tokens=2048,
        repair_max_new_tokens=None,
        max_selected_cells=8,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=True,
        api_concurrency=1,
        api_retry=2,
        api_timeout=120.0,
        api_backoff=2.0,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.strict_json = bool(strict_json)
        self.grid_size = int(grid_size)
        self.image_size = int(image_size)
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.repair_max_new_tokens = int(repair_max_new_tokens) if repair_max_new_tokens is not None else max(self.max_new_tokens, 4096)
        self.max_selected_cells = int(max_selected_cells)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.enable_thinking = bool(enable_thinking)
        self.api_concurrency = int(api_concurrency)
        self.api_retry = max(1, int(api_retry))
        self.api_timeout = float(api_timeout)
        self.api_backoff = float(api_backoff)
        self._client = None
        self._capability_checked = False
        self._capability_result = None

    def _resolved_base_url(self):
        override = os.environ.get(self.base_url_env)
        if override:
            return _normalize_base_url(override)
        return _normalize_base_url(self.base_url)

    def _resolved_api_key(self):
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)

    def _load_client(self):
        if self._client is not None:
            return self._client
        api_key = self._resolved_api_key()
        if not api_key:
            raise EnvironmentError(f"{self.api_key_env} environment variable is not set")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OfoxVLMEvaluator requires the openai Python package") from exc
        self._client = OpenAI(
            base_url=self._resolved_base_url(),
            api_key=api_key,
            timeout=self.api_timeout,
        )
        return self._client

    def _system_prompt(self):
        return (
            "You are a strict multimodal semantic evaluator for ASCR Stage 2. "
            "You must inspect the attached image, reason about prompt-image semantic fidelity, "
            "and return valid JSON only."
        )

    def _build_question(self, original_prompt):
        labels = ", ".join(chr(65 + row) + str(col + 1) for row in range(self.grid_size) for col in range(self.grid_size))
        parts = [
            "You are the external VLM teacher for ASCR Stage 2.",
            "The image contains visible 4x4 grid labels. Treat grid lines and labels as localization aids, not scene content.",
            f"Original text-to-image prompt: {original_prompt}",
            f"Grid cells are {labels}. Rows A-D go top to bottom and columns 1-4 go left to right.",
            "Decide whether the image materially violates the prompt. Check objects, counts, colors, attributes, text, and spatial relations.",
            "Return exactly one valid JSON object. Start with { and end with }.",
            "Do not include markdown, code fences, or bullet points.",
            'Use this schema: {"has_error": boolean, "summary": string, "regions": array, "correction_instruction": string}.',
            "If there is no material semantic error, set has_error to false and regions to an empty array.",
            f"If there is an error, choose at most {self.max_selected_cells} smallest 4x4 grid cells that tightly cover the wrong, missing, or extra content.",
            "Each region must include cells, reason, confidence, error_type, and action set to reopen.",
        ]
        if self.enable_thinking:
            parts.insert(6, "The assistant response may start inside a <think> block, but after that it must write FINAL_JSON: followed immediately by exactly one valid JSON object.")
            parts.insert(7, "Stop immediately after the final JSON object closing }.")
        else:
            parts.insert(0, "/no_think")
            parts.insert(6, "Do not include prose or analysis.")
        return " ".join(parts)

    def _build_json_repair_question(self, original_prompt, previous_text):
        previous_text = " ".join(str(previous_text).split())[:2000]
        return " ".join([
            "/no_think",
            "Return one compact valid JSON object for this semantic evaluation.",
            "Your entire response must be only the JSON object and must end with }.",
            f"Original text-to-image prompt: {original_prompt}",
            "The attached image is the same 4x4-grid overlay image from the failed diagnosis; inspect it again before answering.",
            "Do not add prose, markdown, code fences, or analysis.",
            "Use this schema: {\"has_error\": boolean, \"summary\": string, \"regions\": array, \"correction_instruction\": string}.",
            "Keep summary under 25 words and each reason under 10 words.",
            "If the image satisfies the prompt, return has_error false and regions [].",
            f"If there is an error, select at most {self.max_selected_cells} total grid cells across all regions; each region needs cells, reason, confidence, error_type, and action set to reopen.",
            "Previous malformed or incomplete response:",
            previous_text,
        ])

    def _build_capability_check_question(self):
        return " ".join([
            "/no_think",
            "This is a multimodal capability check.",
            "Inspect the attached image and return exactly one JSON object with no extra prose.",
            'Use this schema: {"vision_ok": boolean, "can_see_image": boolean, "summary": string}.',
            "Set both booleans to true only if you can actually inspect the attached image.",
        ])

    def _messages(self, image_path, question):
        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path), "detail": "high"}},
                ],
            },
        ]

    def _response_metadata(self, response, latency_seconds, stage, fallback_without_max_input_tokens):
        choices = getattr(response, "choices", None) or []
        finish_reason = None
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
        return {
            "stage": stage,
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "finish_reason": finish_reason,
            "usage": _usage_from_response(response),
            "latency_seconds": latency_seconds,
            "fallback_without_max_input_tokens": bool(fallback_without_max_input_tokens),
        }

    def _create_completion(self, messages, max_new_tokens, stage):
        client = self._load_client()
        last_error = None
        for attempt in range(self.api_retry):
            try:
                started = time.perf_counter()
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=int(max_new_tokens),
                        temperature=self.temperature,
                        top_p=self.top_p,
                        extra_body={"max_input_tokens": self.max_input_tokens} if self.max_input_tokens > 0 else None,
                    )
                    fallback_without_max_input_tokens = False
                except Exception:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=int(max_new_tokens),
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    fallback_without_max_input_tokens = True
                latency_seconds = time.perf_counter() - started
                return _response_text(response), self._response_metadata(
                    response,
                    latency_seconds=latency_seconds,
                    stage=stage,
                    fallback_without_max_input_tokens=fallback_without_max_input_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.api_retry:
                    time.sleep(self.api_backoff * (attempt + 1))
        raise RuntimeError(f"Ofox API call failed after {self.api_retry} attempts: {last_error}")

    def _ensure_capability_check(self, image_path):
        if self._capability_checked:
            return
        text, metadata = self._create_completion(
            self._messages(image_path, self._build_capability_check_question()),
            max_new_tokens=min(self.max_new_tokens, 128),
            stage="capability_check",
        )
        try:
            payload = _extract_json_object(text)
        except Exception as exc:
            raise RuntimeError(f"multimodal capability check returned malformed JSON: {exc}") from exc
        if not bool(payload.get("vision_ok")) or not bool(payload.get("can_see_image", payload.get("vision_ok"))):
            raise RuntimeError(f"model {self.model} did not confirm image-input capability: {payload}")
        self._capability_result = {
            "raw_text": text,
            "payload": payload,
            "metadata": metadata,
        }
        self._capability_checked = True

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f"Missing image for Ofox VLM evaluation: {grid_image_path}")
        diagnosis_text = ""
        diagnosis_meta = None
        repair_text = ""
        repair_meta = None
        payload = None
        try:
            self._ensure_capability_check(grid_image_path)
            diagnosis_text, diagnosis_meta = self._create_completion(
                self._messages(grid_image_path, self._build_question(original_prompt)),
                max_new_tokens=self.max_new_tokens,
                stage="diagnosis",
            )
            json_text = diagnosis_text
            try:
                raw_payload = _extract_json_object(json_text)
            except Exception:
                if not self.strict_json:
                    raise
                repair_text, repair_meta = self._create_completion(
                    self._messages(grid_image_path, self._build_json_repair_question(original_prompt, diagnosis_text)),
                    max_new_tokens=self.repair_max_new_tokens,
                    stage="repair",
                )
                json_text = repair_text
                raw_payload = _extract_json_object(json_text)
            payload = _normalize_payload(raw_payload, max_selected_cells=self.max_selected_cells)
            evaluation = parse_semantic_evaluation(payload, grid_size=self.grid_size, max_selected_cells=self.max_selected_cells)
        except Exception as exc:
            return SemanticEvaluation.abstain(
                f"Ofox VLM evaluator failed: {exc}",
                raw={
                    "backend": "ofox_vlm",
                    "model": self.model,
                    "base_url": self._resolved_base_url(),
                    "capability_check": self._capability_result,
                    "ofox_raw_text": diagnosis_text,
                    "ofox_repair_text": repair_text or None,
                    "diagnosis_api": diagnosis_meta,
                    "repair_api": repair_meta,
                },
            )
        raw = {
            "backend": "ofox_vlm",
            "model": self.model,
            "base_url": self._resolved_base_url(),
            "capability_check": self._capability_result,
            "ofox_raw_text": diagnosis_text,
            "ofox_payload": payload,
            "ofox_repair_text": repair_text or None,
            "diagnosis_api": diagnosis_meta,
            "repair_api": repair_meta,
            "api_settings": {
                "max_input_tokens": self.max_input_tokens,
                "max_new_tokens": self.max_new_tokens,
                "repair_max_new_tokens": self.repair_max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "api_concurrency": self.api_concurrency,
                "api_retry": self.api_retry,
                "api_timeout": self.api_timeout,
                "api_backoff": self.api_backoff,
            },
        }
        return SemanticEvaluation(
            has_error=evaluation.has_error,
            summary=evaluation.summary,
            regions=evaluation.regions,
            correction_instruction=evaluation.correction_instruction,
            should_abstain=evaluation.should_abstain,
            parser_error=evaluation.parser_error,
            raw=raw,
        )

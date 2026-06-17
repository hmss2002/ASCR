import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import io
import json
import os
from pathlib import Path
import threading

from ascr.core.schemas import safe_parse_semantic_evaluation
from ascr.distill.api_client import DEFAULT_BASE_URL, DEFAULT_MODEL, api_settings, build_client, chat_completion_text


PROTOCOL = "ascr.api_teacher_distill.v2"
COMPACT_JSON_INSTRUCTION = (
    "Return only one compact JSON object. No analysis. No markdown. No code fences. "
    "No thinking text. Start with { and end with }."
)


def _json_candidates(text):
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                yield candidate
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
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    yield cleaned[start:index + 1]
                    break


def extract_json_object(text):
    parsed = []
    last_error = None
    for candidate in _json_candidates(text):
        try:
            parsed.append(json.loads(candidate))
        except json.JSONDecodeError as exc:
            last_error = exc
    if parsed:
        schema_keys = {"has_error", "baseline_score", "final_score"}
        for payload in reversed(parsed):
            if isinstance(payload, dict) and schema_keys.intersection(payload):
                return payload
        return parsed[-1]
    if last_error is not None:
        raise ValueError(f"response contained JSON-like text but no valid object: {last_error}")
    raise ValueError("response did not contain a JSON object")


def encode_image_data_url(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }[suffix]
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to encode non-PNG/JPEG images for API teacher calls") from exc
    with Image.open(path) as img:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def resolve_path(path, project_root):
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(project_root) / resolved
    return resolved.resolve()


def portable_path(path, project_root, out_root=None, mode="relative"):
    if path is None:
        return None
    resolved = Path(path).resolve()
    if mode == "absolute":
        return str(resolved)
    bases = []
    if out_root is not None:
        bases.append(Path(out_root).resolve())
    bases.append(Path(project_root).resolve())
    for base in bases:
        try:
            return str(resolved.relative_to(base)).replace("\\", "/")
        except ValueError:
            continue
    return str(resolved)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_stage1_records(out_root, limit=None):
    records_dir = Path(out_root) / "records"
    paths = sorted(records_dir.glob("p*.json"))
    if limit is not None:
        paths = paths[: int(limit)]
    for path in paths:
        payload = load_json(path)
        if payload.get("error"):
            continue
        yield path, payload


def find_trace_path(out_root, idx):
    run_root = Path(out_root) / "runs" / f"p{int(idx):03d}"
    traces = sorted(run_root.glob("*/trace.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return traces[0] if traces else None


def read_trace_records(trace_path):
    if not trace_path or not Path(trace_path).exists():
        return []
    records = []
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def localization_messages(prompt, grid_image_path, grid_size=4, max_selected_cells=6, compact=True):
    labels = ", ".join(chr(65 + row) + str(col + 1) for row in range(int(grid_size)) for col in range(int(grid_size)))
    text = (
        f"{COMPACT_JSON_INSTRUCTION}\n"
        f"Original prompt: {prompt}\n"
        f"Grid cells: {labels}. Rows go top to bottom; columns go left to right. "
        "Grid lines and labels are aids, not scene content. "
        "Judge material prompt-image errors. Schema: "
        '{"has_error": boolean, "summary": string, "regions": array, "correction_instruction": string}. '
        f"If error exists, choose at most {int(max_selected_cells)} cells total. "
        'Each region needs cells, reason, confidence, error_type, action="reopen". '
        "If no error, use has_error=false and regions=[]."
    )
    if not compact:
        text += " Explain nothing outside JSON."
    return [
        {"role": "system", "content": "You are an ASCR semantic localization teacher."},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": encode_image_data_url(grid_image_path), "detail": "high"}},
        ]},
    ]


def quality_messages(prompt, baseline_image, final_image, compact=True):
    text = (
        f"{COMPACT_JSON_INSTRUCTION}\n"
        f"Prompt: {prompt}\n"
        "Image A is baseline. Image B is ASCR final. Score prompt following from 0.0 to 1.0. "
        "Choose winner as baseline, final, or tie. Schema: "
        '{"baseline_score": number, "final_score": number, "winner": string, "reason": string}.'
    )
    if not compact:
        text += " Explain nothing outside JSON."
    return [
        {"role": "system", "content": "You are a strict text-to-image quality teacher."},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": encode_image_data_url(baseline_image), "detail": "high"}},
            {"type": "image_url", "image_url": {"url": encode_image_data_url(final_image), "detail": "high"}},
        ]},
    ]


def normalize_quality(payload):
    baseline_score = float(payload.get("baseline_score", payload.get("score_a", 0.0)))
    final_score = float(payload.get("final_score", payload.get("score_b", 0.0)))
    baseline_score = max(0.0, min(1.0, baseline_score))
    final_score = max(0.0, min(1.0, final_score))
    winner = str(payload.get("winner", "")).lower().strip()
    if winner in {"b", "ascr", "self", "final_image"}:
        winner = "final"
    elif winner in {"a", "base", "baseline_image"}:
        winner = "baseline"
    if winner not in {"baseline", "final", "tie"}:
        if abs(final_score - baseline_score) < 0.05:
            winner = "tie"
        else:
            winner = "final" if final_score > baseline_score else "baseline"
    return {
        "baseline_score": baseline_score,
        "final_score": final_score,
        "winner": winner,
        "reason": str(payload.get("reason", ""))[:1000],
    }


def completed_ids(jsonl_path):
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        sample_id = payload.get("sample_id")
        if sample_id:
            done.add(sample_id)
    return done


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, payload):
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")


def build_tasks(out_root, project_root, limit=None, path_mode="relative"):
    out_root = Path(out_root).resolve()
    project_root = Path(project_root).resolve()
    tasks = []
    for rec_path, record in iter_stage1_records(out_root, limit=limit):
        idx = int(record.get("idx", rec_path.stem.lstrip("p")))
        prompt = record.get("prompt", "")
        trace_path = find_trace_path(out_root, idx)
        traces = read_trace_records(trace_path)
        baseline_image = resolve_path(record.get("baseline_image"), project_root)
        final_image = resolve_path(record.get("final_image"), project_root)
        tasks.append({
            "kind": "quality",
            "sample_id": f"p{idx:03d}",
            "idx": idx,
            "prompt": prompt,
            "record_path": portable_path(rec_path, project_root, out_root, path_mode),
            "baseline_image": portable_path(baseline_image, project_root, out_root, path_mode),
            "final_image": portable_path(final_image, project_root, out_root, path_mode),
            "_baseline_image_abs": str(baseline_image) if baseline_image else None,
            "_final_image_abs": str(final_image) if final_image else None,
        })
        for trace in traces:
            artifact_paths = trace.get("artifact_paths", {})
            grid_image = resolve_path(artifact_paths.get("grid_image"), project_root)
            tasks.append({
                "kind": "localization",
                "sample_id": f"p{idx:03d}:i{int(trace.get('iteration', 0)):03d}",
                "idx": idx,
                "iteration": int(trace.get("iteration", 0)),
                "prompt": trace.get("original_prompt", prompt),
                "current_prompt": trace.get("current_prompt"),
                "record_path": portable_path(rec_path, project_root, out_root, path_mode),
                "trace_path": portable_path(trace_path, project_root, out_root, path_mode) if trace_path else None,
                "grid_image": portable_path(grid_image, project_root, out_root, path_mode),
                "_grid_image_abs": str(grid_image) if grid_image else None,
            })
    return tasks


def public_task(task):
    return {key: value for key, value in task.items() if not key.startswith("_")}


def error_payload(task, exc, model, raw_text=None):
    payload = {**public_task(task), "error": str(exc), "teacher_model": model}
    if raw_text:
        payload["raw_preview"] = str(raw_text)[:1000]
    return payload


def run_task(task, client, model, grid_size, max_selected_cells, retries, quality_max_tokens=2048, localization_max_tokens=2048, include_raw_text=False, compact_prompt=True):
    if task["kind"] == "quality":
        baseline_image = task.get("_baseline_image_abs") or task.get("baseline_image")
        final_image = task.get("_final_image_abs") or task.get("final_image")
        for label, path in (("baseline_image", baseline_image), ("final_image", final_image)):
            if not path or not Path(path).exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
        raw = chat_completion_text(
            client,
            quality_messages(task["prompt"], baseline_image, final_image, compact=compact_prompt),
            model=model,
            max_tokens=quality_max_tokens,
            retries=retries,
        )
        parsed = normalize_quality(extract_json_object(raw))
        result = {**public_task(task), "teacher_model": model, "quality": parsed}
        if include_raw_text:
            result["raw_text"] = raw
        return result
    grid_image = task.get("_grid_image_abs") or task.get("grid_image")
    if not grid_image or not Path(grid_image).exists():
        raise FileNotFoundError(f"Missing grid_image: {grid_image}")
    raw = chat_completion_text(
        client,
        localization_messages(task["prompt"], grid_image, grid_size=grid_size, max_selected_cells=max_selected_cells, compact=compact_prompt),
        model=model,
        max_tokens=localization_max_tokens,
        retries=retries,
    )
    payload = extract_json_object(raw)
    evaluation = safe_parse_semantic_evaluation(payload, grid_size=grid_size, max_selected_cells=max_selected_cells)
    result = {**public_task(task), "teacher_model": model, "evaluation": evaluation.to_dict()}
    if include_raw_text:
        result["raw_text"] = raw
    return result


def run_distill(args):
    project_root = Path(args.project_root).resolve()
    out_root = resolve_path(args.out_root, project_root)
    output_dir = resolve_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = api_settings()
    model = args.model or settings["model"]
    client = build_client(base_url=args.base_url or settings["base_url"])

    localization_path = output_dir / "localization_labels.jsonl"
    quality_path = output_dir / "quality_labels.jsonl"
    errors_path = output_dir / "errors.jsonl"
    localization_done = completed_ids(localization_path)
    quality_done = completed_ids(quality_path)
    localization_writer = JsonlWriter(localization_path)
    quality_writer = JsonlWriter(quality_path)
    error_writer = JsonlWriter(errors_path)

    tasks = build_tasks(out_root, project_root, limit=args.limit, path_mode=args.path_mode)
    pending = []
    for task in tasks:
        if task["kind"] == "quality" and task["sample_id"] in quality_done:
            continue
        if task["kind"] == "localization" and task["sample_id"] in localization_done:
            continue
        pending.append(task)

    counts = {"quality": 0, "localization": 0, "errors": 0, "skipped_existing": len(tasks) - len(pending)}

    def handle(task):
        return run_task(
            task,
            client,
            model,
            args.grid_size,
            args.max_selected_cells,
            args.retries,
            quality_max_tokens=args.quality_max_tokens,
            localization_max_tokens=args.localization_max_tokens,
            include_raw_text=args.include_raw_text,
            compact_prompt=not args.no_compact_prompt,
        )

    if args.workers <= 1:
        for task in pending:
            try:
                result = handle(task)
            except Exception as exc:
                counts["errors"] += 1
                error_writer.write(error_payload(task, exc, model))
                continue
            if result["kind"] == "quality":
                quality_writer.write(result)
                counts["quality"] += 1
            else:
                localization_writer.write(result)
                counts["localization"] += 1
    else:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {executor.submit(handle, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    counts["errors"] += 1
                    error_writer.write(error_payload(task, exc, model))
                    continue
                if result["kind"] == "quality":
                    quality_writer.write(result)
                    counts["quality"] += 1
                else:
                    localization_writer.write(result)
                    counts["localization"] += 1

    manifest = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "out_root": portable_path(out_root, project_root, out_root, args.path_mode),
        "output_dir": portable_path(output_dir, project_root, out_root, args.path_mode),
        "base_url": args.base_url or settings["base_url"],
        "teacher_model": model,
        "limit": args.limit,
        "workers": args.workers,
        "grid_size": args.grid_size,
        "max_selected_cells": args.max_selected_cells,
        "quality_max_tokens": args.quality_max_tokens,
        "localization_max_tokens": args.localization_max_tokens,
        "path_mode": args.path_mode,
        "include_raw_text": bool(args.include_raw_text),
        "compact_prompt": not args.no_compact_prompt,
        "counts": counts,
        "files": {
            "localization_labels": portable_path(localization_path, project_root, out_root, args.path_mode),
            "quality_labels": portable_path(quality_path, project_root, out_root, args.path_mode),
            "errors": portable_path(errors_path, project_root, out_root, args.path_mode),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Generate ASCR API teacher distillation labels from Stage-1 outputs.")
    parser.add_argument("--out-root", default="outputs/lumina_qwen_hard64")
    parser.add_argument("--output-dir", default="outputs/teacher_distill/hard64_lumina_qwen")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default=None, help=f"Teacher model (default env ASCR_TEACHER_MODEL or {DEFAULT_MODEL}).")
    parser.add_argument("--base-url", default=None, help=f"OpenAI-compatible base URL (default env OFOX_BASE_URL or {DEFAULT_BASE_URL}).")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-selected-cells", type=int, default=6)
    parser.add_argument("--quality-max-tokens", type=int, default=int(os.environ.get("ASCR_TEACHER_QUALITY_MAX_TOKENS", "2048")))
    parser.add_argument("--localization-max-tokens", type=int, default=int(os.environ.get("ASCR_TEACHER_LOCALIZATION_MAX_TOKENS", "2048")))
    parser.add_argument("--path-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--include-raw-text", action="store_true")
    parser.add_argument("--no-compact-prompt", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = run_distill(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["counts"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

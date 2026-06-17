import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import io
import json
from pathlib import Path
import re
import threading

from ascr.core.schemas import safe_parse_semantic_evaluation
from ascr.distill.api_client import DEFAULT_BASE_URL, DEFAULT_MODEL, api_settings, build_client, chat_completion_text


PROTOCOL = "ascr.api_teacher_distill.v1"


def extract_json_object(text):
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(cleaned)


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


def localization_messages(prompt, grid_image_path, grid_size=4, max_selected_cells=6):
    labels = ", ".join(chr(65 + row) + str(col + 1) for row in range(int(grid_size)) for col in range(int(grid_size)))
    return [
        {
            "role": "system",
            "content": (
                "You are an ASCR teacher model. Judge semantic prompt-image consistency and localize only material errors. "
                "Return compact JSON only."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f'Original prompt: "{prompt}"\n'
                    f"Grid cells: {labels}. Rows go top to bottom; columns go left to right.\n"
                    "The grid and labels are evaluation aids, not scene content.\n"
                    "Return exactly one JSON object with schema: "
                    '{"has_error": boolean, "summary": string, "regions": array, "correction_instruction": string}. '
                    f"If there is an error, choose at most {int(max_selected_cells)} cells total. "
                    "Each region must include cells, reason, confidence, error_type, and action=\"reopen\". "
                    "If the image satisfies the prompt, use has_error=false and regions=[]."
                )},
                {"type": "image_url", "image_url": {"url": encode_image_data_url(grid_image_path), "detail": "high"}},
            ],
        },
    ]


def quality_messages(prompt, baseline_image, final_image):
    return [
        {
            "role": "system",
            "content": (
                "You are a strict text-to-image teacher judge. Compare prompt following for two images. "
                "Return compact JSON only."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f'Prompt: "{prompt}"\n'
                    "Image A is the baseline. Image B is the ASCR final image. "
                    "Score each image from 0.0 to 1.0 for prompt following, then choose winner as "
                    '"baseline", "final", or "tie". Return exactly this JSON schema: '
                    '{"baseline_score": number, "final_score": number, "winner": string, "reason": string}.'
                )},
                {"type": "image_url", "image_url": {"url": encode_image_data_url(baseline_image), "detail": "high"}},
                {"type": "image_url", "image_url": {"url": encode_image_data_url(final_image), "detail": "high"}},
            ],
        },
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


def build_tasks(out_root, project_root, limit=None):
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
            "record_path": str(rec_path),
            "baseline_image": str(baseline_image) if baseline_image else None,
            "final_image": str(final_image) if final_image else None,
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
                "record_path": str(rec_path),
                "trace_path": str(trace_path) if trace_path else None,
                "grid_image": str(grid_image) if grid_image else None,
            })
    return tasks


def run_task(task, client, model, grid_size, max_selected_cells, retries):
    if task["kind"] == "quality":
        for key in ("baseline_image", "final_image"):
            if not task.get(key) or not Path(task[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {task.get(key)}")
        raw = chat_completion_text(
            client,
            quality_messages(task["prompt"], task["baseline_image"], task["final_image"]),
            model=model,
            max_tokens=512,
            retries=retries,
        )
        parsed = normalize_quality(extract_json_object(raw))
        return {**task, "teacher_model": model, "raw_text": raw, "quality": parsed}
    if not task.get("grid_image") or not Path(task["grid_image"]).exists():
        raise FileNotFoundError(f"Missing grid_image: {task.get('grid_image')}")
    raw = chat_completion_text(
        client,
        localization_messages(task["prompt"], task["grid_image"], grid_size=grid_size, max_selected_cells=max_selected_cells),
        model=model,
        max_tokens=1024,
        retries=retries,
    )
    payload = extract_json_object(raw)
    evaluation = safe_parse_semantic_evaluation(payload, grid_size=grid_size, max_selected_cells=max_selected_cells)
    return {**task, "teacher_model": model, "raw_text": raw, "evaluation": evaluation.to_dict()}


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

    tasks = build_tasks(out_root, project_root, limit=args.limit)
    pending = []
    for task in tasks:
        if task["kind"] == "quality" and task["sample_id"] in quality_done:
            continue
        if task["kind"] == "localization" and task["sample_id"] in localization_done:
            continue
        pending.append(task)

    counts = {"quality": 0, "localization": 0, "errors": 0, "skipped_existing": len(tasks) - len(pending)}

    def handle(task):
        return run_task(task, client, model, args.grid_size, args.max_selected_cells, args.retries)

    if args.workers <= 1:
        for task in pending:
            try:
                result = handle(task)
            except Exception as exc:
                counts["errors"] += 1
                error_writer.write({**task, "error": str(exc), "teacher_model": model})
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
                    error_writer.write({**task, "error": str(exc), "teacher_model": model})
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
        "out_root": str(out_root),
        "output_dir": str(output_dir),
        "base_url": args.base_url or settings["base_url"],
        "teacher_model": model,
        "limit": args.limit,
        "workers": args.workers,
        "grid_size": args.grid_size,
        "max_selected_cells": args.max_selected_cells,
        "counts": counts,
        "files": {
            "localization_labels": str(localization_path),
            "quality_labels": str(quality_path),
            "errors": str(errors_path),
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
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = run_distill(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["counts"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


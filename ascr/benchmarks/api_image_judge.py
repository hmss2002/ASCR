import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from ascr.distill.api_client import DEFAULT_MODEL, api_settings, build_client, chat_completion_text
from ascr.distill.teacher import COMPACT_JSON_INSTRUCTION, encode_image_data_url, extract_json_object_with_repair


QUALITY_SCHEMA_TEXT = '{"before_score": number, "after_score": number, "winner": string, "reason": string}'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def completed_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id:
            done.add(sample_id)
    return done


def resolve_path(path, manifest_path):
    raw = Path(path)
    if raw.is_absolute():
        return raw
    candidates = [
        Path.cwd() / raw,
        Path(manifest_path).resolve().parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def judge_messages(prompt, before_image, after_image):
    text = (
        f"{COMPACT_JSON_INSTRUCTION}\n"
        f"Prompt: {prompt}\n"
        "Image A is before distillation: the initial generator image. "
        "Image B is after distillation: the final ASCR candidate produced by the evaluator-selector reopen loop. "
        "Score prompt following from 0.0 to 1.0 for each image. "
        "Choose winner as before, after, or tie. Schema: "
        f"{QUALITY_SCHEMA_TEXT}."
    )
    return [
        {"role": "system", "content": "You are a strict text-to-image before/after quality judge."},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": encode_image_data_url(before_image), "detail": "high"}},
            {"type": "image_url", "image_url": {"url": encode_image_data_url(after_image), "detail": "high"}},
        ]},
    ]


def normalize_judgment(payload):
    before_score = float(payload.get("before_score", payload.get("baseline_score", payload.get("score_a", 0.0))))
    after_score = float(payload.get("after_score", payload.get("final_score", payload.get("score_b", 0.0))))
    before_score = max(0.0, min(1.0, before_score))
    after_score = max(0.0, min(1.0, after_score))
    winner = str(payload.get("winner", "")).lower().strip()
    if winner in {"a", "baseline", "base", "before_image"}:
        winner = "before"
    elif winner in {"b", "final", "ascr", "after_image"}:
        winner = "after"
    if winner not in {"before", "after", "tie"}:
        if abs(after_score - before_score) < 0.05:
            winner = "tie"
        else:
            winner = "after" if after_score > before_score else "before"
    return {
        "before_score": before_score,
        "after_score": after_score,
        "winner": winner,
        "reason": str(payload.get("reason", ""))[:1000],
    }


def dedupe_rows_by_sample_id(rows):
    deduped = []
    seen = set()
    for row in reversed(rows):
        sample_id = row.get("sample_id")
        key = sample_id or json.dumps(row, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    deduped.reverse()
    return deduped


def rewrite_latest_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = dedupe_rows_by_sample_id(read_jsonl(path))
    write_jsonl(path, rows)
    return rows


def prune_resolved_errors(path, resolved_ids):
    path = Path(path)
    if not path.exists():
        return []
    rows = [row for row in read_jsonl(path) if row.get("sample_id") not in resolved_ids]
    rows = dedupe_rows_by_sample_id(rows)
    write_jsonl(path, rows)
    return rows


def summarize(rows, errors, output_dir, manifest, model):
    winners = {"before": 0, "after": 0, "tie": 0}
    before_scores = []
    after_scores = []
    for row in rows:
        judgment = row.get("judgment", {})
        winner = judgment.get("winner")
        if winner in winners:
            winners[winner] += 1
        before_scores.append(float(judgment.get("before_score", 0.0)))
        after_scores.append(float(judgment.get("after_score", 0.0)))
    summary = {
        "schema_version": "ascr.api_image_judge.summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "manifest": str(manifest),
        "model": model,
        "row_count": len(rows),
        "error_count": len(errors),
        "winners": winners,
        "mean_before_score": sum(before_scores) / len(before_scores) if before_scores else None,
        "mean_after_score": sum(after_scores) / len(after_scores) if after_scores else None,
        "mean_delta_after_minus_before": (sum(after_scores) - sum(before_scores)) / len(after_scores) if after_scores else None,
        "output_dir": str(output_dir),
    }
    Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_judge(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "judgments.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if args.overwrite:
        for path in (judgments_path, errors_path):
            if path.exists():
                path.unlink()
    settings = api_settings()
    model = args.model or settings["model"] or DEFAULT_MODEL
    max_tokens = int(args.max_tokens or os.environ.get("ASCR_TEACHER_QUALITY_MAX_TOKENS", 2048))
    repair_retries = int(args.repair_retries or os.environ.get("ASCR_TEACHER_JSON_REPAIR_RETRIES", 1))
    client = build_client(base_url=args.base_url or settings["base_url"])
    rows = read_jsonl(args.manifest)
    done = completed_ids(judgments_path)
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id in done and not args.overwrite:
            continue
        try:
            before_image = resolve_path(row["before_image"], args.manifest)
            after_image = resolve_path(row["after_image"], args.manifest)
            raw_text = chat_completion_text(
                client,
                judge_messages(row.get("prompt", ""), before_image, after_image),
                model=model,
                max_tokens=max_tokens,
                retries=args.retries,
            )
            payload = extract_json_object_with_repair(raw_text, client, model, QUALITY_SCHEMA_TEXT, repair_retries=repair_retries)
            judgment = normalize_judgment(payload)
            result = {
                "schema_version": "ascr.api_image_judge.v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "sample_id": sample_id,
                "domain": row.get("domain"),
                "prompt": row.get("prompt"),
                "before_image": row.get("before_image"),
                "after_image": row.get("after_image"),
                "student_model": row.get("student_model"),
                "evaluator_backend": row.get("evaluator_backend"),
                "model": model,
                "judgment": judgment,
            }
            if args.include_raw_text:
                result["raw_text"] = raw_text
            append_jsonl(judgments_path, result)
        except Exception as exc:
            append_jsonl(errors_path, {
                "sample_id": sample_id,
                "domain": row.get("domain"),
                "prompt": row.get("prompt"),
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:1000],
            })
            if not args.keep_going:
                raise
    judgments = rewrite_latest_rows(judgments_path)
    resolved_ids = {row.get("sample_id") for row in judgments if row.get("sample_id")}
    errors = prune_resolved_errors(errors_path, resolved_ids)
    return summarize(judgments, errors, output_dir, args.manifest, model)


def build_parser():
    parser = argparse.ArgumentParser(description="Judge before/after ASCR image benchmark outputs with Qwen3.7/OFOX.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--repair-retries", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-raw-text", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = run_judge(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

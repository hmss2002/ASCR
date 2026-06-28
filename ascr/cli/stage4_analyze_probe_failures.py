"""Analyze Stage-4 MMU probe failures without loading Lumina."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from ascr.distill.teacher import extract_json_object
from ascr.training.stage4_mmu_lora import mmu_localization_prompt


CELL_TEXT_RE = re.compile(r"\b[A-Z]{1,2}\d+\b|[A-Z]_\d+_\d+x\d+|cell[_-]?\d+", re.IGNORECASE)


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    rows = []
    if not path:
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _payload_from_row(row):
    payload = row.get("parsed_payload")
    if isinstance(payload, dict):
        return payload, None
    raw_text = row.get("raw_text") or ""
    try:
        parsed = extract_json_object(raw_text)
    except Exception as exc:
        return None, str(exc)
    return parsed if isinstance(parsed, dict) else None, None


def _expected_cell_key(grid_size):
    return f"corrupted_cells_{int(grid_size)}x{int(grid_size)}"


def _classify_probe_row(row):
    status = str(row.get("status") or "")
    raw_text = str(row.get("raw_text") or "")
    grid_size = int(row.get("grid_size") or 16)
    expected_key = _expected_cell_key(grid_size)
    target = {str(value) for value in row.get("target_cells", []) or []}
    predicted = {str(value) for value in row.get("predicted_cells", []) or []}
    if status == "call_error":
        return "call_error"
    if not raw_text.strip():
        return "empty_output"
    payload, parse_error = _payload_from_row(row)
    if payload is None:
        if raw_text.strip().startswith("{") or "}" in raw_text:
            return "invalid_json_object"
        if CELL_TEXT_RE.search(raw_text):
            return "non_json_cell_label_text"
        return "no_json_object"
    keys = {str(key) for key in payload.keys()}
    lowered = {key.lower().replace(" ", "_") for key in keys}
    if expected_key in payload:
        value = payload.get(expected_key)
        if not isinstance(value, list):
            return "malformed_localization_value"
        if status == "parsed" and target.intersection(predicted):
            return "parsed_hit"
        if status == "parsed":
            return "valid_format_wrong_cells"
        return "expected_key_but_parser_abstained"
    if "regions" in payload:
        if status == "parsed" and target.intersection(predicted):
            return "legacy_semantic_evaluation_hit"
        return "legacy_semantic_evaluation_no_hit"
    if "has_cells" in lowered or "hascells" in lowered:
        return "wrong_key_has_cells"
    if any(key.startswith("cell") for key in lowered):
        return "wrong_key_cell_key"
    if "corrupted_cells" in payload:
        return "generic_corrupted_cells_key"
    if any(CELL_TEXT_RE.search(str(value)) for value in payload.values()):
        return "cell_labels_under_wrong_key"
    if parse_error:
        return "invalid_json_object"
    return "schema_key_mismatch"


def _index_sft_examples(rows):
    by_sample = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id:
            by_sample[sample_id].append(row)
    return by_sample


def _prompt_alignment(row, sft_rows):
    if not sft_rows:
        return {"sft_match": None, "matched_sft_examples": 0}
    expected = mmu_localization_prompt(
        row.get("prompt", ""),
        grid_size=int(row.get("grid_size") or 16),
        max_selected_cells=int(row.get("max_selected_cells") or 16),
        target_schema=row.get("target_schema") or "localization_cells",
    )
    exact = sum(1 for item in sft_rows if item.get("input_text") == expected)
    modes = sorted({str(item.get("input_mode")) for item in sft_rows if item.get("input_mode")})
    schemas = sorted({str(item.get("target_schema")) for item in sft_rows if item.get("target_schema")})
    return {
        "sft_match": exact > 0,
        "matched_sft_examples": len(sft_rows),
        "matching_prompt_examples": exact,
        "sft_input_modes": modes,
        "sft_target_schemas": schemas,
    }


def _summarize_training_targets(train_rows):
    if not train_rows:
        return None
    key_counts = Counter()
    parse_errors = 0
    unique_answers = set()
    for row in train_rows:
        answer = str(row.get("answer_text") or "")
        unique_answers.add(answer)
        try:
            payload = json.loads(answer)
        except Exception:
            parse_errors += 1
            continue
        if isinstance(payload, dict):
            for key in payload:
                key_counts[str(key)] += 1
    return {
        "row_count": len(train_rows),
        "unique_answer_text_count": len(unique_answers),
        "answer_parse_error_count": parse_errors,
        "answer_key_counts": dict(sorted(key_counts.items())),
    }


def analyze_probe_failures(probe_rows_path, summary_path=None, sft_examples_path=None, train_jsonl_path=None):
    probe_rows = _read_jsonl(probe_rows_path)
    summary = _read_json(summary_path) if summary_path else None
    sft_by_sample = _index_sft_examples(_read_jsonl(sft_examples_path))
    train_target_summary = _summarize_training_targets(_read_jsonl(train_jsonl_path))
    enriched = []
    counts = Counter()
    status_counts = Counter()
    prompt_alignment_counts = Counter()
    examples = defaultdict(list)
    for index, row in enumerate(probe_rows):
        classification = _classify_probe_row(row)
        sample_id = str(row.get("sample_id") or "")
        alignment = _prompt_alignment(row, sft_by_sample.get(sample_id, []))
        if alignment["sft_match"] is not None:
            prompt_alignment_counts[str(alignment["sft_match"]).lower()] += 1
        target = [str(value) for value in row.get("target_cells", []) or []]
        predicted = [str(value) for value in row.get("predicted_cells", []) or []]
        hit_any = bool(set(target).intersection(predicted))
        payload, _ = _payload_from_row(row)
        enriched_row = {
            "row_index": index,
            "sample_id": sample_id,
            "status": row.get("status"),
            "classification": classification,
            "grid_size": row.get("grid_size"),
            "input_mode": row.get("input_mode"),
            "target_schema": row.get("target_schema"),
            "target_cells": target,
            "predicted_cells": predicted,
            "hit_any": hit_any,
            "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            "raw_preview": str(row.get("raw_text") or "")[:500],
            "prompt_alignment": alignment,
        }
        enriched.append(enriched_row)
        counts[classification] += 1
        status_counts[str(row.get("status") or "unknown")] += 1
        if len(examples[classification]) < 3:
            examples[classification].append(enriched_row)
    return {
        "schema_version": "ascr.stage4.probe_failure_analysis.v1",
        "created_at_utc": _created_at(),
        "probe_rows": str(probe_rows_path),
        "summary_path": str(summary_path) if summary_path else None,
        "sft_examples": str(sft_examples_path) if sft_examples_path else None,
        "train_jsonl": str(train_jsonl_path) if train_jsonl_path else None,
        "probe_summary": summary,
        "row_count": len(probe_rows),
        "classification_counts": dict(sorted(counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "prompt_alignment_counts": dict(sorted(prompt_alignment_counts.items())),
        "training_targets": train_target_summary,
        "examples": {key: value for key, value in sorted(examples.items())},
        "rows": enriched,
    }


def _fmt(value):
    if value is None:
        return ""
    return str(value)


def write_outputs(output_dir, analysis):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "failure_summary.json"
    rows_path = output_dir / "failure_rows.jsonl"
    markdown_path = output_dir / "failure_summary.md"
    compact = {key: value for key, value in analysis.items() if key != "rows"}
    summary_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(rows_path, analysis["rows"])
    lines = [
        "# Stage-4 Probe Failure Analysis",
        "",
        f"Probe rows: `{analysis['probe_rows']}`",
        f"Rows analyzed: {analysis['row_count']}",
        "",
        "## Failure Classes",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    for label, count in analysis["classification_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Prompt Alignment", "", "| SFT prompt match | Count |", "| --- | ---: |"])
    for label, count in analysis["prompt_alignment_counts"].items():
        lines.append(f"| {label} | {count} |")
    if analysis.get("training_targets"):
        target_summary = analysis["training_targets"]
        lines.extend([
            "",
            "## Training Targets",
            "",
            f"- Rows: {_fmt(target_summary.get('row_count'))}",
            f"- Unique answer_text values: {_fmt(target_summary.get('unique_answer_text_count'))}",
            f"- answer_text parse errors: {_fmt(target_summary.get('answer_parse_error_count'))}",
            f"- Answer keys: `{json.dumps(target_summary.get('answer_key_counts', {}), sort_keys=True)}`",
        ])
    lines.extend(["", "## Example Rows", ""])
    for label, rows in analysis["examples"].items():
        lines.append(f"### {label}")
        lines.append("")
        for row in rows:
            lines.append(
                "- `{sample}` status={status} target={target} predicted={predicted} keys={keys} raw=`{raw}`".format(
                    sample=row["sample_id"],
                    status=row["status"],
                    target=",".join(row["target_cells"]),
                    predicted=",".join(row["predicted_cells"]),
                    keys=",".join(row["payload_keys"]),
                    raw=row["raw_preview"].replace("`", "'").replace("\n", "\\n")[:160],
                )
            )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "failure_summary": str(summary_path),
        "failure_rows": str(rows_path),
        "failure_markdown": str(markdown_path),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Classify Stage-4 MMU localization probe failures.")
    parser.add_argument("--probe-rows", required=True, help="Path to probe_rows.jsonl.")
    parser.add_argument("--summary", default=None, help="Optional probe summary.json.")
    parser.add_argument("--sft-examples", default=None, help="Optional Stage-4 sft_examples.jsonl or train_sft_examples.jsonl.")
    parser.add_argument("--train-jsonl", default=None, help="Optional converted Lumina train.jsonl.")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    analysis = analyze_probe_failures(
        args.probe_rows,
        summary_path=args.summary,
        sft_examples_path=args.sft_examples,
        train_jsonl_path=args.train_jsonl,
    )
    outputs = write_outputs(args.output_dir, analysis)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

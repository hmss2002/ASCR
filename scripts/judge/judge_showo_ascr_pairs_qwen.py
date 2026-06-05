#!/usr/bin/env python
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ascr.core.config import load_config
from ascr.evaluators.qwen_vl import _extract_json_object
from ascr.evaluators.registry import build_evaluator


def resolve_path(value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path.cwd() / path


def latest_path(root, pattern):
    candidates = sorted(Path(root).rglob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_comparison_results(input_path):
    path = Path(input_path)
    if path.is_dir():
        suite_path = path / "suite.json"
        if not suite_path.exists():
            suite_path = latest_path(path, "suite.json")
        if suite_path is not None:
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            return suite.get("results", []), suite_path
        comparison_paths = sorted(path.rglob("comparison.json"), key=lambda item: item.stat().st_mtime)
        results = []
        for comparison_path in comparison_paths:
            item = json.loads(comparison_path.read_text(encoding="utf-8"))
            item.setdefault("result_path", str(comparison_path))
            results.append(item)
        return results, path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "suite.json" or "results" in payload:
        return payload.get("results", []), path
    payload.setdefault("result_path", str(path))
    return [payload], path


def clean_judge_question(prompt):
    return " ".join([
        "/no_think",
        "You are a strict text-to-image prompt-following judge.",
        "Evaluate the clean generated image only; there are no grid labels or diagnostic overlays.",
        f"Prompt: {prompt}",
        "Check objects, counts, colors, attributes, text, style, and spatial relations.",
        "Return exactly one compact JSON object and no prose.",
        "Schema: {\"matches_prompt\": boolean, \"score\": number, \"summary\": string, \"missing_or_wrong\": array}.",
        "Set matches_prompt true only if the important semantic requirements are satisfied.",
        "Use score from 0.0 to 1.0. Keep summary under 25 words.",
    ])


def repair_question(prompt, previous_text):
    previous_text = " ".join(str(previous_text).split())[:1800]
    return " ".join([
        "/no_think",
        "Return exactly one valid JSON object for this clean-image prompt-following judgment.",
        f"Prompt: {prompt}",
        "Schema: {\"matches_prompt\": boolean, \"score\": number, \"summary\": string, \"missing_or_wrong\": array}.",
        "Do not include markdown or prose. Previous response:",
        previous_text,
    ])


def normalize_payload(payload):
    matches = payload.get("matches_prompt", payload.get("match", payload.get("is_match", False)))
    try:
        score = float(payload.get("score", 1.0 if matches else 0.0))
    except Exception:
        score = 1.0 if matches else 0.0
    score = max(0.0, min(1.0, score))
    missing = payload.get("missing_or_wrong", payload.get("errors", [])) or []
    if isinstance(missing, str):
        missing = [missing]
    return {
        "matches_prompt": bool(matches),
        "score": score,
        "summary": str(payload.get("summary", ""))[:500],
        "missing_or_wrong": missing,
    }


def judge_image(evaluator, prompt, image_path, pass_threshold, include_raw):
    raw_text = ""
    json_text = ""
    try:
        raw_text = evaluator._generate_text(clean_judge_question(prompt), str(image_path), enable_thinking=False)
        json_text = raw_text
        try:
            payload = _extract_json_object(raw_text)
        except Exception:
            json_text = evaluator._generate_text(
                repair_question(prompt, raw_text),
                str(image_path),
                enable_thinking=False,
                max_new_tokens=getattr(evaluator, "repair_max_new_tokens", 768),
            )
            payload = _extract_json_object(json_text)
        payload = normalize_payload(payload)
        status = "pass" if payload["matches_prompt"] and payload["score"] >= pass_threshold else "fail"
        result = {"status": status, "image": str(image_path), "payload": payload}
    except Exception as exc:
        result = {"status": "abstain", "image": str(image_path), "error": str(exc)}
    if include_raw:
        result["raw_text"] = raw_text
        if json_text and json_text != raw_text:
            result["repair_text"] = json_text
    return result


def pair_verdict(baseline_status, ascr_status):
    if "abstain" in {baseline_status, ascr_status}:
        return "judge_abstain"
    if baseline_status == "fail" and ascr_status == "pass":
        return "ascr_win"
    if baseline_status == "pass" and ascr_status == "fail":
        return "ascr_loss"
    if baseline_status == "pass" and ascr_status == "pass":
        return "both_pass"
    return "both_fail"


def score_of(record):
    return float(record.get("payload", {}).get("score", 0.0))


def write_markdown(path, report):
    lines = [
        "| Index | Prompt | Baseline | ASCR | Pair verdict | Baseline score | ASCR score |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for index, record in enumerate(report["records"]):
        prompt = record["prompt"].replace("|", " ")
        lines.append("| {} | {} | {} | {} | {} | {:.3f} | {:.3f} |".format(
            index,
            prompt,
            record["baseline"]["status"],
            record["ascr"]["status"],
            record["pair_verdict"],
            score_of(record["baseline"]),
            score_of(record["ascr"]),
        ))
    lines.append("")
    lines.append("Counts: " + json.dumps(report["counts"], sort_keys=True))
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Judge clean baseline-vs-ASCR final image pairs with Qwen.")
    parser.add_argument("input_path", help="suite.json, comparison.json, or an output directory containing them")
    parser.add_argument("--config", default="configs/stage1/showo/stage1_showo_qwen35_9b_fullcap_parallel.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--repair-max-new-tokens", type=int, default=256)
    parser.add_argument("--pass-threshold", type=float, default=0.5)
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    evaluator_config = config.setdefault("evaluator", {})
    if args.max_new_tokens is not None:
        evaluator_config["max_new_tokens"] = int(args.max_new_tokens)
    if args.repair_max_new_tokens is not None:
        evaluator_config["repair_max_new_tokens"] = int(args.repair_max_new_tokens)
    results, source_path = load_comparison_results(args.input_path)
    if args.limit is not None:
        results = results[:max(0, int(args.limit))]
    if not results:
        raise ValueError("No comparison results found for final judge")

    evaluator = build_evaluator(config.get("evaluator", {}).get("name", "qwen_vl"), config)
    records = []
    counts = Counter()
    for index, result in enumerate(results):
        prompt = result["prompt"]
        baseline_image = resolve_path(result.get("baseline_image"))
        ascr_image = resolve_path(result.get("ascr_final_image") or result.get("final_decoded_image"))
        if baseline_image is None or ascr_image is None:
            raise ValueError("Result is missing baseline_image and clean ASCR final image")
        print(json.dumps({"event": "judge_clean_pair_start", "index": index, "prompt": prompt}), flush=True)
        baseline = judge_image(evaluator, prompt, baseline_image, args.pass_threshold, args.include_raw)
        print(json.dumps({"event": "judge_clean_image", "index": index, "side": "baseline", "status": baseline["status"]}), flush=True)
        ascr = judge_image(evaluator, prompt, ascr_image, args.pass_threshold, args.include_raw)
        print(json.dumps({"event": "judge_clean_image", "index": index, "side": "ascr", "status": ascr["status"]}), flush=True)
        verdict = pair_verdict(baseline["status"], ascr["status"])
        counts[verdict] += 1
        counts["baseline_" + baseline["status"]] += 1
        counts["ascr_" + ascr["status"]] += 1
        records.append({
            "prompt": prompt,
            "result_path": result.get("result_path"),
            "heuristic_comparison": result.get("comparison", {}),
            "baseline": baseline,
            "ascr": ascr,
            "pair_verdict": verdict,
        })
        print(json.dumps({"event": "judged_clean_pair", "index": index, "pair_verdict": verdict, "baseline": baseline["status"], "ascr": ascr["status"]}), flush=True)

    report = {
        "protocol": "qwen_clean_final_pair_judge_v1",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_path": str(source_path),
        "config": args.config,
        "pass_threshold": args.pass_threshold,
        "prompt_count": len(records),
        "counts": dict(counts),
        "caveat": "This judges clean final images without ASCR grid overlays. Qwen3.5-9B is also the ASCR loop evaluator, so treat this as automated benchmark signal, not independent human evidence.",
        "records": records,
    }
    output_path = Path(args.output) if args.output else (Path(source_path).parent if Path(source_path).is_file() else Path(source_path)) / "qwen_clean_final_pair_judge.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + chr(10), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    write_markdown(markdown_path, report)
    print(json.dumps({"output_path": str(output_path), "markdown_path": str(markdown_path), "counts": dict(counts)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

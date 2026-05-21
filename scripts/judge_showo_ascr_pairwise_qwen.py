#!/usr/bin/env python
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

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

def make_pair_image(baseline_image, ascr_image, output_path, baseline_label="baseline", ascr_label="ASCR", show_labels=True):
    left = Image.open(baseline_image).convert("RGB")
    right = Image.open(ascr_image).convert("RGB")
    height = max(left.height, right.height)
    width = left.width + right.width
    label_height = 32 if show_labels else 0
    canvas = Image.new("RGB", (width, height + label_height), "white")
    canvas.paste(left, (0, label_height))
    canvas.paste(right, (left.width, label_height))
    if show_labels:
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"LEFT: {baseline_label}", fill=(0, 0, 0))
        draw.text((left.width + 12, 8), f"RIGHT: {ascr_label}", fill=(0, 0, 0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    left.close()
    right.close()
    return output_path

def pairwise_question(prompt, baseline_label="baseline", ascr_label="ASCR"):
    return " ".join([
        "/no_think",
        "You are a strict text-to-image prompt-following pairwise judge.",
        "The image contains two clean generated images placed side by side.",
        f"LEFT is {baseline_label}. RIGHT is {ascr_label}.",
        f"Prompt: {prompt}",
        "Check objects, counts, colors, attributes, text, and spatial relations.",
        "Choose which side better satisfies the prompt. Tie only if neither side is materially better.",
        "Return exactly one compact JSON object and no prose.",
        "Schema: {\"winner\": \"baseline|ascr|tie\", \"confidence\": number, \"summary\": string, \"baseline_errors\": array, \"ascr_errors\": array}.",
    ])

def repair_question(prompt, previous_text):
    previous_text = " ".join(str(previous_text).split())[:1800]
    return " ".join([
        "/no_think",
        "Return exactly one valid JSON object for this pairwise prompt-following judgment.",
        f"Prompt: {prompt}",
        "Schema: {\"winner\": \"baseline|ascr|tie\", \"confidence\": number, \"summary\": string, \"baseline_errors\": array, \"ascr_errors\": array}.",
        "Do not include markdown or prose. Previous response:",
        previous_text,
    ])

def normalize_pairwise_payload(payload):
    winner = str(payload.get("winner", payload.get("choice", "tie"))).strip().lower()
    if winner in {"left", "base", "baseline_image"}:
        winner = "baseline"
    elif winner in {"right", "candidate", "ascr_image"}:
        winner = "ascr"
    elif winner not in {"baseline", "ascr", "tie"}:
        winner = "tie"
    try:
        confidence = float(payload.get("confidence", payload.get("score", 0.5)))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return {
        "winner": winner,
        "confidence": confidence,
        "summary": str(payload.get("summary", ""))[:500],
        "baseline_errors": payload.get("baseline_errors", []),
        "ascr_errors": payload.get("ascr_errors", []),
    }

def judge_pair(evaluator, prompt, baseline_image, ascr_image, pair_image, include_raw, baseline_label="baseline", ascr_label="ASCR", show_image_labels=True):
    raw_text = ""
    json_text = ""
    try:
        make_pair_image(baseline_image, ascr_image, pair_image, baseline_label, ascr_label, show_image_labels)
        raw_text = evaluator._generate_text(pairwise_question(prompt, baseline_label, ascr_label), str(pair_image), enable_thinking=False)
        json_text = raw_text
        try:
            payload = _extract_json_object(raw_text)
        except Exception:
            json_text = evaluator._generate_text(
                repair_question(prompt, raw_text),
                str(pair_image),
                enable_thinking=False,
                max_new_tokens=getattr(evaluator, "repair_max_new_tokens", 768),
            )
            payload = _extract_json_object(json_text)
        result = {"status": "pass", "pair_image": str(pair_image), "payload": normalize_pairwise_payload(payload)}
    except Exception as exc:
        result = {"status": "abstain", "pair_image": str(pair_image), "error": str(exc)}
    if include_raw:
        result["raw_text"] = raw_text
        if json_text and json_text != raw_text:
            result["repair_text"] = json_text
    return result

def pairwise_verdict(pairwise):
    if pairwise.get("status") == "abstain":
        return "judge_abstain"
    winner = pairwise.get("payload", {}).get("winner", "tie")
    if winner == "ascr":
        return "ascr_win"
    if winner == "baseline":
        return "ascr_loss"
    return "pairwise_tie"

def score_of(pairwise):
    return float(pairwise.get("payload", {}).get("confidence", 0.0))

def write_markdown(path, report):
    lines = [
        "| Index | Prompt | Winner | Verdict | Confidence | Summary |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for index, record in enumerate(report["records"]):
        prompt = record["prompt"].replace("|", " ")
        payload = record["pairwise"].get("payload", {})
        lines.append("| {} | {} | {} | {} | {:.3f} | {} |".format(
            index,
            prompt,
            payload.get("winner", "abstain"),
            record["pairwise_verdict"],
            score_of(record["pairwise"]),
            str(payload.get("summary", record["pairwise"].get("error", ""))).replace("|", " "),
        ))
    lines.append("")
    lines.append("Counts: " + json.dumps(report["counts"], sort_keys=True))
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Judge clean baseline-vs-ASCR final image pairs with side-by-side Qwen pairwise comparison.")
    parser.add_argument("input_path", help="suite.json, comparison.json, or an output directory containing them")
    parser.add_argument("--config", default="configs/stage1_showo_qwen35_9b_fullcap_parallel.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--repair-max-new-tokens", type=int, default=256)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--ascr-label", default="ASCR")
    parser.add_argument("--no-image-labels", action="store_true", help="Do not draw LEFT/RIGHT labels into the paired image canvas.")
    parser.add_argument("--shard-id", type=int, default=0, help="0-based shard index (default 0).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (default 1 = no sharding).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    evaluator_config = config.setdefault("evaluator", {})
    evaluator_config["max_new_tokens"] = int(args.max_new_tokens)
    evaluator_config["repair_max_new_tokens"] = int(args.repair_max_new_tokens)
    results, source_path = load_comparison_results(args.input_path)
    if args.limit is not None:
        results = results[:max(0, int(args.limit))]
    if not results:
        raise ValueError("No comparison results found for pairwise judge")

    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"Invalid shard config: shard-id={args.shard_id} num-shards={args.num_shards}")
    sharded = args.num_shards > 1
    if sharded:
        # Round-robin assignment so workload is balanced even if prompts differ in cost.
        original_total = len(results)
        results = [r for i, r in enumerate(results) if i % args.num_shards == args.shard_id]
        print(json.dumps({"event": "shard_info", "shard_id": args.shard_id, "num_shards": args.num_shards, "shard_size": len(results), "total": original_total}), flush=True)

    output_path = Path(args.output) if args.output else (Path(source_path).parent if Path(source_path).is_file() else Path(source_path)) / "qwen_pairwise_judge.json"
    pair_dir = output_path.with_suffix("") / "pairwise_images"
    evaluator = build_evaluator(config.get("evaluator", {}).get("name", "qwen_vl"), config)
    records = []
    counts = Counter()
    for shard_local_index, result in enumerate(results):
        index = shard_local_index * args.num_shards + args.shard_id if sharded else shard_local_index
        prompt = result["prompt"]
        baseline_image = resolve_path(result.get("baseline_image"))
        ascr_image = resolve_path(result.get("accepted_ascr_image") or result.get("ascr_final_image") or result.get("final_decoded_image"))
        if baseline_image is None or ascr_image is None:
            raise ValueError("Result is missing baseline_image and clean ASCR final image")
        pair_image = pair_dir / f"pair_{index:03d}.png"
        print(json.dumps({"event": "judge_pairwise_start", "index": index, "prompt": prompt}), flush=True)
        pairwise = judge_pair(evaluator, prompt, baseline_image, ascr_image, pair_image, args.include_raw, args.baseline_label, args.ascr_label, not args.no_image_labels)
        verdict = pairwise_verdict(pairwise)
        counts[verdict] += 1
        records.append({
            "prompt": prompt,
            "result_path": result.get("result_path"),
            "baseline_image": str(baseline_image),
            "ascr_image": str(ascr_image),
            "heuristic_comparison": result.get("heuristic_comparison", result.get("comparison", {})),
            "pairwise": pairwise,
            "pairwise_verdict": verdict,
        })
        print(json.dumps({"event": "judged_pairwise", "index": index, "pairwise_verdict": verdict}), flush=True)

    report = {
        "protocol": "qwen_clean_side_by_side_pairwise_judge_v1",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_path": str(source_path),
        "config": args.config,
        "prompt_count": len(records),
        "counts": dict(counts),
        "baseline_label": args.baseline_label,
        "ascr_label": args.ascr_label,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + chr(10), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    write_markdown(markdown_path, report)
    print(json.dumps({"output_path": str(output_path), "markdown_path": str(markdown_path), "counts": dict(counts)}, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

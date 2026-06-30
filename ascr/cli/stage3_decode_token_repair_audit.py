import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from ascr.analysis.stage3_token_repair import read_json, read_jsonl, write_jsonl
from ascr.generators.lumina_native import LuminaNativeEngine


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def select_audit_rows(rows, max_pairs=16, seed=0):
    positives = [
        row
        for row in rows
        if row.get("row_type") == "positive"
        or row.get("target_json", {}).get("error")
        or bool(row.get("target_json", {}).get("cells"))
    ]
    rng = random.Random(int(seed))
    by_bucket = {}
    for row in positives:
        bucket = (row.get("corruption_mask_size"), row.get("corruption_operator"))
        by_bucket.setdefault(bucket, []).append(row)
    selected = []
    for bucket in sorted(by_bucket, key=lambda item: (str(item[0]), str(item[1]))):
        group = list(by_bucket[bucket])
        rng.shuffle(group)
        if group and len(selected) < int(max_pairs):
            selected.append(group[0])
    if len(selected) < int(max_pairs):
        remaining_ids = {row.get("sample_id") for row in selected}
        remaining = [row for row in positives if row.get("sample_id") not in remaining_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, int(max_pairs) - len(selected))])
    return selected[: int(max_pairs)]


def decode_audit_pairs(
    dataset,
    output_dir,
    max_pairs=16,
    seed=0,
    repo_path="third_party/Lumina-DiMOO",
    checkpoint_path="models/lumina-dimoo",
    device="cuda",
    image_size=1024,
):
    rows = read_jsonl(dataset)
    selected = select_audit_rows(rows, max_pairs=max_pairs, seed=seed)
    if not selected:
        raise ValueError(f"No positive rows available for audit decode in {dataset}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = LuminaNativeEngine(
        checkpoint_path=checkpoint_path,
        repo_path=repo_path,
        device=device,
        image_size=image_size,
    )
    audit_rows = []
    for index, row in enumerate(selected):
        sample_id = _safe_name(row.get("sample_id") or f"row_{index:04d}")
        clean_tokens = read_json(row["clean_vq_ids_path"])
        corrupted_tokens = read_json(row["corrupted_vq_ids_path"])
        clean_image = output_dir / f"{index:04d}_{sample_id}_clean.png"
        corrupted_image = output_dir / f"{index:04d}_{sample_id}_corrupted.png"
        engine.decode_to(clean_tokens, clean_image)
        engine.decode_to(corrupted_tokens, corrupted_image)
        audit_rows.append({
            "schema_version": "ascr.stage3.token_repair_audit_decode.row.v1",
            "created_at_utc": _created_at(),
            "sample_id": row.get("sample_id"),
            "prompt": row.get("prompt", ""),
            "target_json": row.get("target_json"),
            "corruption_mask_size": row.get("corruption_mask_size"),
            "corruption_operator": row.get("corruption_operator"),
            "clean_image": str(clean_image),
            "corrupted_image": str(corrupted_image),
            "clean_vq_ids_path": row.get("clean_vq_ids_path"),
            "corrupted_vq_ids_path": row.get("corrupted_vq_ids_path"),
        })
    manifest = {
        "schema_version": "ascr.stage3.token_repair_audit_decode.manifest.v1",
        "created_at_utc": _created_at(),
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "row_count": len(audit_rows),
        "seed": int(seed),
        "repo_path": str(repo_path),
        "checkpoint_path": str(checkpoint_path),
        "image_size": int(image_size),
        "audit_manifest": str(output_dir / "audit_manifest.jsonl"),
    }
    write_jsonl(output_dir / "audit_manifest.jsonl", audit_rows)
    (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Decode a tiny sample of Stage-3 token repair pairs for visual audit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-path", default="third_party/Lumina-DiMOO")
    parser.add_argument("--checkpoint-path", default="models/lumina-dimoo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = decode_audit_pairs(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

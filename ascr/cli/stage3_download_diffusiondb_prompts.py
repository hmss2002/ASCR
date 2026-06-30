import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "nsfw", "unsafe"}


def download_diffusiondb_prompts(
    output,
    dataset="poloclub/diffusiondb",
    subset="2m_first_10k",
    split="train",
    limit=20000,
    prompt_field="prompt",
    include_nsfw=False,
    dry_run=False,
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped_nsfw = 0
    if not dry_run:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(
                "Downloading DiffusionDB prompts requires the optional `datasets` package. "
                "Install it on the server with `python -m pip install datasets` or run with --dry-run."
            ) from exc
        stream = load_dataset(dataset, subset, split=split, streaming=True, trust_remote_code=True)
        for index, row in enumerate(stream):
            if len(rows) >= int(limit):
                break
            prompt = str(row.get(prompt_field) or "").strip()
            if not prompt:
                continue
            nsfw_value = row.get("image_nsfw", row.get("nsfw", row.get("safety_concept")))
            if nsfw_value is not None and _truthy(nsfw_value) and not include_nsfw:
                skipped_nsfw += 1
                continue
            rows.append({
                "schema_version": "ascr.stage3.external_prompt.row.v1",
                "source": dataset,
                "subset": subset,
                "split": split,
                "row_index": index,
                "prompt": prompt,
            })
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    manifest = {
        "schema_version": "ascr.stage3.external_prompt.download_manifest.v1",
        "created_at_utc": _created_at(),
        "dataset": dataset,
        "subset": subset,
        "split": split,
        "limit": int(limit),
        "prompt_field": prompt_field,
        "include_nsfw": bool(include_nsfw),
        "dry_run": bool(dry_run),
        "row_count": len(rows),
        "skipped_nsfw": skipped_nsfw,
        "output": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Download prompt-only metadata from DiffusionDB for Stage-3 prompt sourcing.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="poloclub/diffusiondb")
    parser.add_argument("--subset", default="2m_first_10k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--include-nsfw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = download_diffusiondb_prompts(**vars(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def _created_at():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _truthy(value):
    if isinstance(value, (int, float)):
        return float(value) >= 0.5
    return str(value).strip().lower() in {"1", "true", "yes", "y", "nsfw", "unsafe"}


def _cache_dataset_name(dataset):
    return str(dataset).replace("/", "___")


def _candidate_cache_roots():
    roots = []
    if os.environ.get("HF_DATASETS_CACHE"):
        roots.append(Path(os.environ["HF_DATASETS_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "datasets")
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    return roots


def _iter_cached_arrow_rows(dataset, subset):
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception as exc:
        raise RuntimeError(
            "DiffusionDB is cached but this datasets version cannot read it directly. "
            "Install pyarrow or use a newer datasets version."
        ) from exc
    dataset_name = _cache_dataset_name(dataset)
    arrow_paths = []
    for root in _candidate_cache_roots():
        if root.exists():
            arrow_paths.extend(sorted((root / dataset_name).glob(f"{subset}/**/*.arrow")))
    if not arrow_paths:
        searched = ", ".join(str(root / dataset_name / subset) for root in _candidate_cache_roots())
        raise RuntimeError(f"No cached Arrow files found for {dataset}:{subset}. Searched: {searched}")
    for path in arrow_paths:
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                columns = {name: batch.column(name) for name in batch.column_names}
                for index in range(batch.num_rows):
                    yield {name: columns[name][index].as_py() for name in batch.column_names}


def _load_dataset_rows(dataset, subset, split):
    from datasets import load_dataset

    try:
        return load_dataset(dataset, subset, split=split, streaming=True, trust_remote_code=True)
    except ValueError as exc:
        if "trust_remote_code" not in str(exc):
            raise
        try:
            return load_dataset(dataset, subset, split=split, streaming=True)
        except NotImplementedError:
            try:
                return load_dataset(dataset, subset, split=split)
            except NotImplementedError:
                return _iter_cached_arrow_rows(dataset, subset)
    except NotImplementedError:
        try:
            return load_dataset(dataset, subset, split=split, trust_remote_code=True)
        except (NotImplementedError, ValueError):
            return _iter_cached_arrow_rows(dataset, subset)


def download_diffusiondb_prompts(
    output,
    dataset="poloclub/diffusiondb",
    subset="2m_text_only",
    split="train",
    limit=10000,
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
            stream = _load_dataset_rows(dataset, subset, split)
        except Exception as exc:
            raise RuntimeError(
                "Downloading DiffusionDB prompts requires the optional `datasets` package. "
                "Install it on the server with `python -m pip install datasets` or run with --dry-run."
            ) from exc
        for index, row in enumerate(stream):
            if len(rows) >= int(limit):
                break
            prompt = str(row.get(prompt_field) or "").strip()
            if not prompt:
                continue
            nsfw_value = row.get("prompt_nsfw", row.get("image_nsfw", row.get("nsfw", row.get("safety_concept"))))
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
    parser.add_argument("--subset", default="2m_text_only")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10000)
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

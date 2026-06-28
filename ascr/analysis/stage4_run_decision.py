"""Decision helpers for Stage-4 server-side MMU/LoRA runs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


WRONG_FORMAT_CLASSES = {
    "wrong_key_has_cells",
    "wrong_key_cell_key",
    "non_json_cell_label_text",
    "cell_labels_under_wrong_key",
    "schema_key_mismatch",
    "invalid_json_object",
    "generic_corrupted_cells_key",
}


def created_at_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json_files(paths):
    payloads = []
    for path in paths or []:
        candidate = Path(path)
        if candidate.exists():
            payload = read_json(candidate)
            if isinstance(payload, dict):
                payload["_source_path"] = str(candidate)
                payloads.append(payload)
    return payloads


def scan_log_files(paths):
    counts = Counter()
    examples = {}
    for path in paths or []:
        candidate = Path(path)
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lowered = text.lower()
        checks = {
            "cuda_oom": ("cuda out of memory", "torch.outofmemoryerror", "outofmemoryerror", "oom"),
            "qos_submit_limit": ("qosmaxsubmitjobperuserlimit",),
            "gc_not_supported": ("does not support gradient checkpointing",),
            "missing_path": ("no such file or directory", "file not found", "missing ascr environment"),
        }
        for label, needles in checks.items():
            if any(needle in lowered for needle in needles):
                counts[label] += 1
                examples.setdefault(label, str(candidate))
    return {"counts": dict(sorted(counts.items())), "examples": examples}


def _rows(registry, kind=None):
    rows = list((registry or {}).get("rows", []) or [])
    if kind:
        rows = [row for row in rows if row.get("kind") == kind]
    return rows


def _text(row):
    return " ".join(str(row.get(key) or "") for key in ("artifact_id", "path", "lora_path", "summary_path"))


def _as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _gc_active(row):
    backend = row.get("gradient_checkpointing_backend")
    wrapped = row.get("wrapped_module_count")
    if backend == "huggingface":
        return True
    if backend == "ascr_module_wrapper":
        return wrapped is None or int(wrapped) > 0
    return False


def _find_full_gc_adapters(registry):
    return [
        row
        for row in _rows(registry, "lora_adapter")
        if int(row.get("image_size") or 0) >= 1024
        and row.get("gradient_checkpointing")
        and "1024px_gc" in _text(row)
    ]


def _find_grid4_gc_probe(registry):
    candidates = [
        row
        for row in _rows(registry, "probe_summary")
        if int(row.get("grid_size") or 0) == 4 and "1024px_gc" in _text(row)
    ]
    return candidates[-1] if candidates else None


def _dominant_failure_classes(failure_summaries):
    counts = Counter()
    for payload in failure_summaries or []:
        counts.update(payload.get("classification_counts") or {})
    return counts


def _action(kind, priority, title, command=None, reason=None):
    return {
        "kind": kind,
        "priority": int(priority),
        "title": title,
        "command": command,
        "reason": reason,
    }


def decide_stage4_next_actions(
    registry,
    failure_summaries=None,
    log_scan=None,
    parse_threshold=0.5,
    hit_any_threshold=0.01,
):
    failure_counts = _dominant_failure_classes(failure_summaries)
    log_counts = Counter((log_scan or {}).get("counts") or {})
    actions = []
    facts = {
        "registry_rows": len(_rows(registry)),
        "lora_adapters": len(_rows(registry, "lora_adapter")),
        "probe_summaries": len(_rows(registry, "probe_summary")),
        "failure_classification_counts": dict(sorted(failure_counts.items())),
        "log_issue_counts": dict(sorted(log_counts.items())),
    }

    full_gc_adapters = _find_full_gc_adapters(registry)
    active_gc_adapters = [row for row in full_gc_adapters if _gc_active(row)]
    inactive_gc_adapters = [row for row in full_gc_adapters if not _gc_active(row)]
    grid4_probe = _find_grid4_gc_probe(registry)

    if log_counts.get("qos_submit_limit"):
        actions.append(_action(
            "cluster",
            10,
            "Keep Stage-4 arrays small or manually shard jobs.",
            "EPOCHS=1 sbatch --array=0-1 jobs/stage4/train_mmu_lora_gc_probe.sbatch",
            "Slurm logs include QOSMaxSubmitJobPerUserLimit.",
        ))
    if log_counts.get("cuda_oom"):
        if active_gc_adapters:
            actions.append(_action(
                "memory",
                20,
                "GC was active but OOM still appeared; try shorter sequence length or multi-GPU training.",
                "python -m ascr.cli.stage4_train_mmu_lora --config configs/stage4/self_corrupt/mmu_lora_train_hard64_vq_tokens_l40s_1024px_gc_adam8bit.yaml --epochs 1 --max-seq-len 4096",
                "OOM with active gradient checkpointing means single-L40S 6144 tokens may still be too large.",
            ))
        else:
            actions.append(_action(
                "memory",
                20,
                "Confirm gradient checkpointing actually wrapped LLaDA blocks before rerunning memory probes.",
                "cat outputs/stage4_self_corrupt/registry/stage4_run_registry.md",
                "OOM appeared but no active GC adapter is visible in the registry.",
            ))

    if not full_gc_adapters:
        actions.append(_action(
            "run",
            30,
            "Run the two-task 1024px GC smoke probe.",
            "EPOCHS=1 sbatch jobs/stage4/train_mmu_lora_gc_probe.sbatch",
            "No 1024px GC LoRA training manifest is present in the registry.",
        ))
    elif inactive_gc_adapters:
        actions.append(_action(
            "debug",
            30,
            "GC was requested but not proven active; inspect model module names.",
            "python - <<'PY'\nfrom pathlib import Path\nimport sys\nrepo=Path('third_party/Lumina-DiMOO').resolve(); sys.path.insert(0,str(repo))\nfrom model import LLaDAForMultiModalGeneration\nm=LLaDAForMultiModalGeneration.from_pretrained('models/lumina-dimoo', device_map='cpu')\nfor name, module in m.named_modules():\n    cls=module.__class__.__name__\n    if 'Layer' in cls or 'Block' in cls:\n        print(name, cls)\nPY",
            "At least one 1024px GC manifest has no active checkpoint backend or zero wrapped modules.",
        ))
    else:
        best = min(active_gc_adapters, key=lambda row: int(row.get("epochs") or 0))
        if int(best.get("epochs") or 0) <= 1:
            actions.append(_action(
                "scale",
                40,
                "GC smoke fit; run the full hard64 1024px LoRA training.",
                "python -m ascr.cli.stage4_train_mmu_lora --config configs/stage4/self_corrupt/mmu_lora_train_hard64_vq_tokens_l40s_1024px_gc_adam8bit.yaml --epochs 15",
                "A 1024px GC adapter manifest exists with active gradient checkpointing.",
            ))

    if grid4_probe is None:
        if active_gc_adapters:
            actions.append(_action(
                "evaluate",
                50,
                "Evaluate the grid4 1024px GC adapter and analyze failures.",
                "TASK=grid4_1024_gc RUN_PREP=0 RUN_TRAIN=0 RUN_PROBE=1 RUN_ANALYSIS=1 bash scripts/training/run_stage4_gc_probe.sh",
                "No grid4 1024px GC probe summary is present.",
            ))
    else:
        parse_rate = _as_float(grid4_probe.get("parse_rate"))
        hit_any_rate = _as_float(grid4_probe.get("hit_any_rate"))
        facts["grid4_1024_gc_parse_rate"] = parse_rate
        facts["grid4_1024_gc_hit_any_rate"] = hit_any_rate
        wrong_format_count = sum(count for label, count in failure_counts.items() if label in WRONG_FORMAT_CLASSES)
        if parse_rate < float(parse_threshold):
            if wrong_format_count:
                actions.append(_action(
                    "format",
                    60,
                    "Run the prompt/decoding sweep before scaling data.",
                    "sbatch jobs/stage4/stage4_probe_sweep.sbatch\n# after completion:\nMODE=summarize bash scripts/training/run_stage4_probe_sweep.sh\ncat outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_sweep_l40s_1024px_gc/probe_sweep_summary.md",
                    "Failure analysis is dominated by output-format classes; sweep prompt variants and answer length first.",
                ))
            else:
                actions.append(_action(
                    "diagnose",
                    60,
                    "Run failure analysis for the grid4 probe before changing training strategy.",
                    "TASK=grid4_1024_gc RUN_PREP=0 RUN_TRAIN=0 RUN_PROBE=0 RUN_ANALYSIS=1 bash scripts/training/run_stage4_gc_probe.sh",
                    "Grid4 parse rate is below threshold and no failure summary was supplied.",
                ))
        elif hit_any_rate <= float(hit_any_threshold):
            actions.append(_action(
                "localization",
                70,
                "JSON format is acceptable but localization is not; compare predicted-vs-target cells before scaling.",
                "python -m ascr.cli.stage4_analyze_probe_failures --probe-rows outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/probe_rows.jsonl --summary outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/summary.json --output-dir outputs/stage4_self_corrupt/mmu_lora_hard64_curriculum/grid4/vq_tokens/probe_lora_l40s_1024px_gc_eval/failure_analysis",
                "Grid4 parse rate cleared the gate but hit_any is still near zero.",
            ))
        else:
            actions.append(_action(
                "scale",
                80,
                "Grid4 GC probe is useful; scale to grid8/grid16 and Hard256.",
                "PROFILE=l40s GRIDS='8 16' sbatch --array=0-1 jobs/stage4/train_mmu_lora_curriculum.sbatch",
                "Grid4 parse and hit_any rates both cleared the configured gates.",
            ))

    if not actions:
        actions.append(_action(
            "complete",
            100,
            "No immediate Stage-4 blocker detected; build registry and summarize latest outputs.",
            "bash scripts/training/run_stage4_postprocess.sh",
            "Registry and supplied summaries do not require a new training action.",
        ))
    actions.sort(key=lambda action: action["priority"])
    return {
        "schema_version": "ascr.stage4.next_actions.v1",
        "created_at_utc": created_at_utc(),
        "parse_threshold": float(parse_threshold),
        "hit_any_threshold": float(hit_any_threshold),
        "facts": facts,
        "actions": actions,
    }


def write_next_actions(output_dir, decision):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage4_next_actions.json"
    markdown_path = output_dir / "stage4_next_actions.md"
    json_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-4 Next Actions",
        "",
        f"Generated: {decision['created_at_utc']}",
        "",
        "## Facts",
        "",
    ]
    for key, value in sorted((decision.get("facts") or {}).items()):
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(["", "## Actions", ""])
    for action in decision.get("actions", []):
        lines.append(f"### P{action['priority']} {action['title']}")
        if action.get("reason"):
            lines.append("")
            lines.append(action["reason"])
        if action.get("command"):
            lines.append("")
            lines.append("```bash")
            lines.append(action["command"])
            lines.append("```")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"next_actions_json": str(json_path), "next_actions_md": str(markdown_path)}

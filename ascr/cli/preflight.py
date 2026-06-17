import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ascr.core.config import load_config


_OPENAI_KEY_PREFIX = "sk" + "-"
_OFOX_KEY_PREFIX = _OPENAI_KEY_PREFIX + "of" + "-"

SECRET_PATTERNS = [
    ("openai_or_proxy_key", re.compile(r"\b(?:" + re.escape(_OFOX_KEY_PREFIX) + r"|" + re.escape(_OPENAI_KEY_PREFIX) + r")[A-Za-z0-9_-]{16,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("github_token", re.compile(r"\bghp_[0-9A-Za-z]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[0-9A-Za-z]{20,}\b")),
    ("quoted_api_key_assignment", re.compile(r"\b[A-Z0-9_]*API_KEY\s*=\s*['\"][^'\"<][^'\"]{12,}['\"]")),
]

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".sbatch",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _project_root(path=None):
    start = Path(path or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _record(level, message, detail=None):
    item = {"level": level, "message": message}
    if detail is not None:
        item["detail"] = detail
    return item


def _load_config(path):
    if not path:
        return {}
    return load_config(path)


def _nearest_existing_parent(path):
    path = Path(path)
    if path.exists():
        return path
    for parent in path.parents:
        if parent.exists():
            return parent
    return None


def _path_status(path, project_root):
    if not path:
        return None
    path = Path(str(path))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _iter_config_paths(config):
    generator = config.get("generator", {}) if isinstance(config.get("generator"), dict) else {}
    evaluator = config.get("evaluator", {}) if isinstance(config.get("evaluator"), dict) else {}
    generator_name = str(generator.get("name", "")).lower()
    if generator_name == "lumina" and os.environ.get("LUMINA_REPO"):
        yield "generator.repo_path", os.environ["LUMINA_REPO"], "checkout"
    else:
        yield "generator.repo_path", generator.get("repo_path"), "checkout"
    for key in ("checkpoint_path", "vq_model_path", "llm_model_path", "showo_config_path"):
        yield f"generator.{key}", generator.get(key), "file_or_dir"
    if os.environ.get("QWEN_MODEL_PATH"):
        yield "evaluator.model_path", os.environ["QWEN_MODEL_PATH"], "model"
    else:
        yield "evaluator.model_path", evaluator.get("model_path"), "model"


def check_python():
    if sys.version_info < (3, 9):
        return [_record("error", "Python >=3.9 is required", sys.version.split()[0])]
    return [_record("ok", "Python version is supported", sys.version.split()[0])]


def check_imports(mode):
    checks = ["ascr", "ascr.cli.run_stage1", "ascr.core.loop", "ascr.evaluators.registry", "ascr.generators.registry"]
    if mode == "server":
        checks.append("torch")
    records = []
    for module_name in checks:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            level = "error" if module_name != "torch" or mode == "server" else "warn"
            records.append(_record(level, f"Import failed: {module_name}", str(exc)))
        else:
            records.append(_record("ok", f"Import succeeded: {module_name}"))
    return records


def check_cuda(mode):
    try:
        import torch
    except Exception as exc:
        level = "error" if mode == "server" else "warn"
        return [_record(level, "torch is not importable; CUDA could not be checked", str(exc))]
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        in_gpu_context = bool(os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS"))
        nvidia_smi_available = shutil.which("nvidia-smi") is not None
        if mode == "server" and (in_gpu_context or nvidia_smi_available):
            return [_record("error", "CUDA is not available to torch")]
        return [_record("warn", "CUDA is not available to torch in this shell", "Run this check inside a GPU allocation for device validation.")]
    return [_record("ok", "CUDA is available", {"gpu_count": int(torch.cuda.device_count())})]


def check_paths(config, project_root, mode):
    records = []
    for label, raw_path, kind in _iter_config_paths(config):
        if not raw_path:
            continue
        resolved = _path_status(raw_path, project_root)
        if resolved is None:
            continue
        if resolved.exists():
            records.append(_record("ok", f"{label} exists", str(resolved)))
        else:
            level = "error" if mode == "server" and kind in {"checkout", "model", "file_or_dir"} else "warn"
            records.append(_record(level, f"{label} is missing", str(resolved)))
    output_dir = config.get("output_dir", "outputs/stage1")
    parent = _nearest_existing_parent(project_root / str(output_dir))
    if parent and os.access(parent, os.W_OK):
        records.append(_record("ok", "Output path has a writable existing parent", str(parent)))
    else:
        records.append(_record("warn", "Output path parent is not currently writable", str(output_dir)))
    return records


def check_env(required):
    records = []
    for name in required:
        state = "set" if os.environ.get(name) else "unset"
        level = "ok" if state == "set" else "error"
        records.append(_record(level, f"Required env var {name} is {state}"))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"):
        if name in os.environ:
            records.append(_record("ok", f"Runtime env var {name} is set", os.environ.get(name)))
    return records


def _git_ls_files(project_root):
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return [str(p.relative_to(project_root)) for p in project_root.rglob("*") if p.is_file()]
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def scan_secrets(project_root):
    findings = []
    for rel in _git_ls_files(project_root):
        rel_path = Path(rel)
        if rel_path.parts[:2] == ("docs", "archive"):
            continue
        if rel_path.parts and rel_path.parts[0] == ".git":
            continue
        path = project_root / rel_path
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", ".env.template"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            for name, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append({"path": rel, "line": lineno, "pattern": name})
    return findings


def run_preflight(mode, config_path=None, project_root=None, required_env=None, scan_secret_values=False):
    project_root = _project_root(project_root)
    config = _load_config(config_path)
    records = []
    records.extend(check_python())
    records.extend(check_imports(mode))
    records.extend(check_cuda(mode))
    records.extend(check_paths(config, project_root, mode))
    records.extend(check_env(required_env or []))
    if scan_secret_values:
        findings = scan_secrets(project_root)
        if findings:
            records.append(_record("error", "Potential committed secrets found", findings))
        else:
            records.append(_record("ok", "No potential committed secrets found"))
    return {
        "mode": mode,
        "project_root": str(project_root),
        "config": str(config_path) if config_path else None,
        "records": records,
        "ok": not any(item["level"] == "error" for item in records),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Check ASCR local/server readiness without printing secret values.")
    parser.add_argument("--mode", choices=["local", "server"], default="local")
    parser.add_argument("--config", default=None, help="Stage config to inspect for model/checkout/output paths.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--require-env", action="append", default=[], help="Env var that must be set; can be repeated.")
    parser.add_argument("--scan-secrets", action="store_true", help="Scan tracked text files for likely committed secrets.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _print_human(report):
    for item in report["records"]:
        level = item["level"].upper()
        message = item["message"]
        detail = item.get("detail")
        if detail is None:
            print(f"[{level}] {message}")
        else:
            print(f"[{level}] {message}: {detail}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_preflight(
        args.mode,
        config_path=args.config,
        project_root=args.project_root,
        required_env=args.require_env,
        scan_secret_values=args.scan_secrets,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

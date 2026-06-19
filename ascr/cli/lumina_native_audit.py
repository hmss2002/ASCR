import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from ascr.evaluators.lumina_native import (
    ANSWER_IMAGE_METHODS,
    ANSWER_TOKEN_METHODS,
    LuminaNativeEvaluator,
    supported_native_answer_methods,
)
from ascr.generators.lumina_native import LuminaNativeEngine


def _path_status(path):
    value = Path(path) if path else None
    return {"path": str(path) if path else None, "exists": bool(value and value.exists())}


def _source_hits(repo_path):
    repo = Path(repo_path)
    if not repo.exists():
        return []
    needles = [
        "answer_image",
        "evaluate_image",
        "answer_vq_tokens",
        "generate_text",
        "chat",
        "MultiModal",
        "LLaDAForMultiModalGeneration",
    ]
    hits = []
    for path in repo.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        matched = [needle for needle in needles if needle in text]
        if matched:
            hits.append({"file": str(path.relative_to(repo)), "matches": matched})
        if len(hits) >= 32:
            break
    return hits


def run_audit(args):
    repo_path = args.repo_path or os.environ.get("LUMINA_REPO", "third_party/Lumina-DiMOO")
    checkpoint_path = args.checkpoint_path or os.environ.get("LUMINA_MODEL_PATH", "models/lumina-dimoo")
    engine = LuminaNativeEngine(
        checkpoint_path=checkpoint_path,
        repo_path=repo_path,
        device=args.device,
        image_size=args.image_size,
        token_grid_size=args.token_grid_size,
        answer_steps=args.answer_steps,
        answer_block_length=args.answer_block_length,
        answer_temperature=args.answer_temperature,
        answer_cfg_scale=args.answer_cfg_scale,
    )
    methods = supported_native_answer_methods(engine)
    report = {
        "schema_version": "ascr.lumina_native_evaluator_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repo": _path_status(repo_path),
        "checkpoint": _path_status(checkpoint_path),
        "wrapper_class": "ascr.generators.lumina_native.LuminaNativeEngine",
        "required_methods": list(ANSWER_IMAGE_METHODS + ANSWER_TOKEN_METHODS),
        "wrapper_supported_methods": methods,
        "wrapper_supports_native_eval": bool(methods),
        "repo_source_hits": _source_hits(repo_path) if args.scan_repo else [],
        "model_load_attempted": bool(args.load_model),
        "smoke_attempted": bool(args.smoke_image),
        "smoke": None,
        "blocker": None,
    }
    if args.load_model:
        try:
            engine._load()
            report["model_loaded"] = True
        except Exception as exc:
            report["model_loaded"] = False
            report["model_load_error"] = str(exc)
    if args.smoke_image:
        evaluator = LuminaNativeEvaluator(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            device=args.device,
            grid_size=args.grid_size,
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
            max_selected_cells=args.max_selected_cells,
            answer_steps=args.answer_steps,
            answer_block_length=args.answer_block_length,
            answer_temperature=args.answer_temperature,
            answer_cfg_scale=args.answer_cfg_scale,
            engine=engine,
        )
        evaluation = evaluator.evaluate(args.prompt, args.smoke_image, 0)
        report["smoke"] = evaluation.to_dict()
    if not report["wrapper_supports_native_eval"]:
        report["blocker"] = (
            "The current ASCR LuminaNativeEngine exposes generate/decode/reopen only. "
            "A verified image-conditioned text answer method is required before "
            "Lumina-native evaluator distillation can start."
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_supported and not report["wrapper_supports_native_eval"]:
        return 2
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Audit whether Lumina-DiMOO can act as a native ASCR semantic evaluator.")
    parser.add_argument("--repo-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--token-grid-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--max-selected-cells", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--answer-steps", type=int, default=64)
    parser.add_argument("--answer-block-length", type=int, default=128)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--answer-cfg-scale", type=float, default=0.0)
    parser.add_argument("--prompt", default="A red cube left of a blue sphere")
    parser.add_argument("--smoke-image", default=None, help="Optional image/grid path for an evaluator smoke call.")
    parser.add_argument("--scan-repo", action="store_true", help="Search the Lumina checkout for possible MMU/text-output hooks.")
    parser.add_argument("--load-model", action="store_true", help="Actually load the Lumina model. Use only on a GPU server.")
    parser.add_argument("--require-supported", action="store_true", help="Exit nonzero if no native evaluator method is exposed.")
    parser.add_argument("--output", default=None)
    return parser


def main(argv=None):
    return run_audit(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

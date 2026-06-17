import argparse
import json
from pathlib import Path

from ascr.core.config import load_config
from ascr.training.ddp import get_distributed_context
from ascr.training.selector_model import (
    DatasetReplaySelectorModel,
    LearnedCoarseSelectorModel,
    TeacherReplaySelectorModel,
    cell_labels_to_multi_hot,
    load_image_tensor,
    prompt_hash_features,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train a Stage 2 selector baseline from teacher traces or a built Stage 2 dataset.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config with training defaults.")
    parser.add_argument("--dataset", default=None, help="Path to the Stage 2 dataset JSONL file.")
    parser.add_argument("--output-dir", default="checkpoints/stage2_selector_replay", help="Directory for checkpoint metadata.")
    parser.add_argument("--mode", default="dataset_replay", choices=["teacher_replay", "dataset_replay", "learned_coarse"], help="Selector family to materialize.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of dataset samples to read.")
    parser.add_argument("--dry-run", action="store_true", help="Load and summarize the dataset without writing checkpoints.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for learned_coarse training.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for learned_coarse training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for learned_coarse training.")
    parser.add_argument("--device", default=None, help="Training device override for learned_coarse, e.g. cpu or cuda.")
    parser.add_argument("--image-size", type=int, default=64, help="Input image size for learned_coarse training.")
    parser.add_argument("--prompt-hash-dim", type=int, default=256, help="Hashed prompt feature dimension for learned_coarse.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for learned_coarse.")
    parser.add_argument("--cell-threshold", type=float, default=0.5, help="Cell selection threshold stored in learned_coarse metadata.")
    parser.add_argument("--error-threshold", type=float, default=0.5, help="Error threshold stored in learned_coarse metadata.")
    return parser


def _iter_jsonl(path, limit=None):
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def _sample_stats(samples):
    selected_counts = []
    with_errors = 0
    with_after = 0
    with_grid_image = 0
    for sample in samples:
        mask = sample.get("projected_token_mask") or {}
        selected_counts.append(int(mask.get("selected_count", 0)))
        if sample.get("teacher_json", {}).get("has_error"):
            with_errors += 1
        if sample.get("after_image"):
            with_after += 1
        if sample.get("grid_image"):
            with_grid_image += 1
    return {
        "sample_count": len(samples),
        "error_sample_count": with_errors,
        "with_after_image_count": with_after,
        "with_grid_image_count": with_grid_image,
        "avg_selected_tokens": (sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0,
        "max_selected_tokens": max(selected_counts) if selected_counts else 0,
    }


def _training_settings(args, config):
    training = config.get("training", {})
    return {
        "epochs": int(training.get("epochs", args.epochs)),
        "batch_size": int(training.get("batch_size", args.batch_size)),
        "learning_rate": float(training.get("learning_rate", args.learning_rate)),
        "device": training.get("device", args.device),
        "image_size": int(training.get("image_size", args.image_size)),
        "prompt_hash_dim": int(training.get("prompt_hash_dim", args.prompt_hash_dim)),
        "hidden_dim": int(training.get("hidden_dim", args.hidden_dim)),
        "cell_threshold": float(training.get("cell_threshold", args.cell_threshold)),
        "error_threshold": float(training.get("error_threshold", args.error_threshold)),
    }


def _is_primary_process(distributed):
    return int(distributed.get("rank", 0)) == 0


def _select_device(requested_device, distributed):
    if requested_device:
        return requested_device
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return f"cuda:{int(distributed.get('local_rank', 0))}"
    return "cpu"


def _prepare_learned_examples(samples, image_size, prompt_hash_dim, grid_size=4):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("learned_coarse training requires torch") from exc
    examples = []
    skipped = 0
    for sample in samples:
        grid_image = sample.get("grid_image")
        if not grid_image or not Path(grid_image).exists():
            skipped += 1
            continue
        examples.append({
            "image": load_image_tensor(grid_image, image_size=image_size),
            "prompt": torch.tensor(prompt_hash_features(sample.get("prompt", ""), prompt_hash_dim), dtype=torch.float32),
            "iteration": torch.tensor([float(sample.get("iteration", 0))], dtype=torch.float32),
            "cell_targets": torch.tensor(cell_labels_to_multi_hot(sample.get("selected_4x4_cells", []), grid_size=grid_size), dtype=torch.float32),
            "error_target": torch.tensor([1.0 if sample.get("teacher_json", {}).get("has_error") and sample.get("selected_4x4_cells") else 0.0], dtype=torch.float32),
        })
    if not examples:
        raise ValueError("learned_coarse training found no usable samples with existing grid_image paths")
    return examples, skipped


def _batchify(examples, batch_size):
    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start:start + int(batch_size)]
        yield batch


def _stack_batch(batch, device):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("learned_coarse batch stacking requires torch") from exc
    images = torch.stack([item["image"] for item in batch]).to(device)
    prompts = torch.stack([item["prompt"] for item in batch]).to(device)
    iterations = torch.stack([item["iteration"] for item in batch]).to(device)
    cell_targets = torch.stack([item["cell_targets"] for item in batch]).to(device)
    error_targets = torch.stack([item["error_target"] for item in batch]).to(device)
    return images, prompts, iterations, cell_targets, error_targets


def _forward_learned(model, images, prompts, iterations):
    import torch

    features = torch.cat([model.network()[:7](images), prompts, iterations], dim=1)
    logits = model.network()[7:](features)
    return logits[:, :-1], logits[:, -1:]


def _train_learned_coarse(samples, output_dir, distributed, settings):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("learned_coarse training requires torch") from exc
    output_dir = Path(output_dir)
    device = _select_device(settings["device"], distributed)
    examples, skipped = _prepare_learned_examples(
        samples,
        image_size=settings["image_size"],
        prompt_hash_dim=settings["prompt_hash_dim"],
    )
    model = LearnedCoarseSelectorModel(
        prompt_hash_dim=settings["prompt_hash_dim"],
        image_size=settings["image_size"],
        hidden_dim=settings["hidden_dim"],
        grid_size=4,
        max_selected_cells=8,
        metadata={
            "cell_threshold": settings["cell_threshold"],
            "error_threshold": settings["error_threshold"],
            "notes": "Lightweight learned Stage 2 coarse selector baseline trained from teacher traces.",
        },
    )
    network = model.network().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"])
    cell_loss_fn = torch.nn.BCEWithLogitsLoss()
    error_loss_fn = torch.nn.BCEWithLogitsLoss()
    train_batches = max(1, (len(examples) + int(settings["batch_size"]) - 1) // int(settings["batch_size"]))
    history = []
    for epoch in range(settings["epochs"]):
        network.train()
        total_loss = 0.0
        batch_count = 0
        for batch in _batchify(examples, settings["batch_size"]):
            images, prompts, iterations, cell_targets, error_targets = _stack_batch(batch, device)
            cell_logits, error_logits = _forward_learned(model, images, prompts, iterations)
            loss = cell_loss_fn(cell_logits, cell_targets) + error_loss_fn(error_logits, error_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            batch_count += 1
        history.append({"epoch": epoch, "loss": total_loss / max(batch_count, 1)})
    network.eval()
    with torch.inference_mode():
        train_loss = 0.0
        for batch in _batchify(examples, settings["batch_size"]):
            images, prompts, iterations, cell_targets, error_targets = _stack_batch(batch, device)
            cell_logits, error_logits = _forward_learned(model, images, prompts, iterations)
            loss = cell_loss_fn(cell_logits, cell_targets) + error_loss_fn(error_logits, error_targets)
            train_loss += float(loss.detach().cpu().item())
    metrics = {
        "epochs": settings["epochs"],
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "device": device,
        "train_batches": train_batches,
        "skipped_missing_grid_images": skipped,
        "final_train_loss": train_loss / train_batches,
        "history": history,
    }
    if not _is_primary_process(distributed):
        return {"skipped_write": True, "metrics": metrics}
    return model.save(output_dir, metrics=metrics)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else {}
    dataset_path = args.dataset or config.get("dataset_path") or config.get("training", {}).get("dataset_path")
    if not dataset_path:
        raise SystemExit("--dataset is required unless dataset_path is provided in the config")
    output_dir = Path(args.output_dir or config.get("output_dir", "checkpoints/stage2_selector_replay"))
    samples = list(_iter_jsonl(dataset_path, limit=args.limit))
    if not samples:
        raise SystemExit(f"No dataset samples found in {dataset_path}")
    distributed = get_distributed_context()
    stats = _sample_stats(samples)
    payload = {
        "status": "ok",
        "mode": args.mode,
        "config": args.config,
        "dataset": str(dataset_path),
        "distributed": distributed,
        "stats": stats,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        payload["training"] = _training_settings(args, config) if args.mode == "learned_coarse" else None
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.mode == "teacher_replay":
        if not _is_primary_process(distributed):
            payload["checkpoint"] = {"skipped_write": True}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        model = TeacherReplaySelectorModel(metadata={
            "dataset_path": str(dataset_path),
            "stats": stats,
            "distributed": distributed,
            "notes": "Teacher-only replay baseline. It reuses the teacher-provided projected token masks rather than learning a student selector.",
        })
        checkpoint_path = model.save(output_dir / "selector_checkpoint.json")
        payload["checkpoint"] = {"checkpoint_path": str(checkpoint_path)}
    elif args.mode == "dataset_replay":
        if not _is_primary_process(distributed):
            payload["checkpoint"] = {"skipped_write": True}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        model = DatasetReplaySelectorModel.from_samples(samples, metadata={
            "dataset_path": str(dataset_path),
            "stats": stats,
            "distributed": distributed,
            "notes": "Dataset-replay Stage 2 baseline. This is the minimum runnable student-side closure before a learned image-conditioned selector is added.",
        })
        payload["checkpoint"] = model.save(output_dir)
    else:
        payload["training"] = _training_settings(args, config)
        payload["checkpoint"] = _train_learned_coarse(samples, output_dir, distributed, payload["training"])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

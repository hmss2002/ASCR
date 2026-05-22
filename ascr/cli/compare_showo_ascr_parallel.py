import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ascr.cli.compare_showo_ascr import build_suite, load_prompts, suite_to_markdown
from ascr.core.config import load_config


def build_parser():
    parser = argparse.ArgumentParser(description="Run Show-o ASCR prompt comparisons in one-process-per-GPU parallel mode.")
    parser.add_argument("--config", default="configs/stage1_showo_qwen35_9b.yaml")
    parser.add_argument("--prompt", default="A red cube left of a blue sphere")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/benchmarks_qwen35_9b_parallel")
    parser.add_argument("--generation-timesteps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--ascr-start-mode", choices=["baseline", "partial"], default=None)
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids. Defaults to CUDA_VISIBLE_DEVICES or all visible CUDA devices.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum concurrent worker processes.")
    parser.add_argument("--repeat-count", type=int, default=1, help="Repeat every prompt with seed offsets. Use 2 with four prompts to fill eight GPUs.")
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_gpus(requested):
    value = requested or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if gpus:
        return gpus
    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    if count > 0:
        return [str(index) for index in range(count)]
    return [""]


def write_yaml(path, config):
    try:
        import yaml
        text = yaml.safe_dump(config, sort_keys=False)
    except Exception:
        text = json.dumps(config, indent=2, sort_keys=False)
    path.write_text(text.rstrip() + chr(10), encoding="utf-8")


def write_prompt(path, prompt):
    path.write_text(prompt.strip() + chr(10), encoding="utf-8")


def base_seed(config):
    generator = config.get("generator", {}) if isinstance(config.get("generator"), dict) else {}
    return int(config.get("seed", generator.get("seed", 1234)))


def task_seed(seed, repeat_index, seed_step):
    return int(seed) + int(repeat_index) * int(seed_step)


def prepare_task_files(args, root, base_config, prompt, prompt_index, repeat_index, task_index, seed):
    prompt_dir = root / "prompts"
    config_dir = root / "configs"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"prompt_{prompt_index:03d}_repeat_{repeat_index:02d}"
    prompt_path = prompt_dir / f"{task_name}.txt"
    config_path = config_dir / f"{task_name}.yaml"
    worker_dir = root / "workers" / f"task_{task_index:03d}_{task_name}"
    write_prompt(prompt_path, prompt)
    task_config = copy.deepcopy(base_config)
    task_config["seed"] = seed
    generator_config = task_config.setdefault("generator", {})
    if isinstance(generator_config, dict):
        generator_config["seed"] = seed
    base_run_name = str(task_config.get("run_name", "stage1_showo_ascr"))
    task_config["run_name"] = f"{base_run_name}_{task_name}"
    task_config["output_dir"] = str(worker_dir / "ascr")
    write_yaml(config_path, task_config)
    command = [
        args.python_bin,
        "-m",
        "ascr.cli.compare_showo_ascr",
        "--config",
        str(config_path),
        "--prompts-file",
        str(prompt_path),
        "--output-dir",
        str(worker_dir),
        "--generation-timesteps",
        str(args.generation_timesteps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--max-iterations",
        str(args.max_iterations),
    ]
    if args.ascr_start_mode:
        command.extend(["--ascr-start-mode", args.ascr_start_mode])
    return {
        "task_index": task_index,
        "prompt_index": prompt_index,
        "repeat_index": repeat_index,
        "seed": seed,
        "prompt": prompt,
        "prompt_path": str(prompt_path),
        "config_path": str(config_path),
        "worker_dir": str(worker_dir),
        "command": command,
    }


def build_tasks(args, root, prompts, base_config):
    repeat_count = max(1, int(args.repeat_count))
    seed = base_seed(base_config)
    tasks = []
    for prompt_index, prompt in enumerate(prompts):
        for repeat_index in range(repeat_count):
            task_index = len(tasks)
            tasks.append(prepare_task_files(args, root, base_config, prompt, prompt_index, repeat_index, task_index, task_seed(seed, repeat_index, args.seed_step)))
    return tasks


def launch_task(task, gpu, logs_dir, project_root):
    task_index = task["task_index"]
    stdout_path = logs_dir / f"task_{task_index:03d}.out"
    stderr_path = logs_dir / f"task_{task_index:03d}.err"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(task["command"], cwd=project_root, env=env, stdout=stdout_handle, stderr=stderr_handle, text=True)
    return {"process": process, "task": task, "gpu": gpu, "stdout_path": stdout_path, "stderr_path": stderr_path, "stdout_handle": stdout_handle, "stderr_handle": stderr_handle}


def finish_task(record):
    record["stdout_handle"].close()
    record["stderr_handle"].close()
    process = record["process"]
    task = record["task"]
    if process.returncode != 0:
        message = "Worker {} on GPU {} failed with exit code {}. See {} and {}.".format(task["task_index"], record["gpu"], process.returncode, record["stdout_path"], record["stderr_path"])
        raise RuntimeError(message)
    worker_dir = Path(task["worker_dir"])
    matches = sorted(worker_dir.rglob("comparison.json"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise RuntimeError("Worker {} finished but produced no comparison.json under {}".format(task["task_index"], worker_dir))
    result_path = matches[-1]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["result_path"] = str(result_path)
    result["markdown_path"] = str(result_path.with_suffix(".md"))
    result["task_index"] = task["task_index"]
    result["prompt_index"] = task["prompt_index"]
    result["repeat_index"] = task["repeat_index"]
    result["seed"] = task["seed"]
    result["worker_gpu"] = record["gpu"]
    result["worker_stdout"] = str(record["stdout_path"])
    result["worker_stderr"] = str(record["stderr_path"])
    return result


def run_tasks(tasks, gpus, max_workers, logs_dir, project_root):
    worker_count = min(len(tasks), len(gpus), max_workers or len(gpus))
    if worker_count <= 0:
        raise ValueError("No worker slots are available")
    pending = list(tasks)
    running = []
    available_gpus = list(gpus[:worker_count])
    results = []
    try:
        while pending or running:
            while pending and available_gpus:
                gpu = available_gpus.pop(0)
                task = pending.pop(0)
                print(json.dumps({"event": "launch", "task_index": task["task_index"], "gpu": gpu, "prompt_index": task["prompt_index"], "repeat_index": task["repeat_index"], "seed": task["seed"]}), flush=True)
                running.append(launch_task(task, gpu, logs_dir, project_root))
            finished_any = False
            for record in list(running):
                return_code = record["process"].poll()
                if return_code is None:
                    continue
                running.remove(record)
                record["process"].returncode = return_code
                result = finish_task(record)
                results.append(result)
                available_gpus.append(record["gpu"])
                print(json.dumps({"event": "finish", "task_index": result["task_index"], "gpu": result["worker_gpu"], "verdict": result["comparison"].get("verdict")}), flush=True)
                finished_any = True
            if not finished_any and running:
                time.sleep(1)
    except Exception:
        for record in running:
            try:
                record["process"].terminate()
            except Exception:
                pass
            try:
                record["stdout_handle"].close()
                record["stderr_handle"].close()
            except Exception:
                pass
        raise
    return sorted(results, key=lambda item: item["task_index"])


def main(argv=None):
    args = build_parser().parse_args(argv)
    project_root = Path.cwd()
    base_config = load_config(args.config)
    prompts = load_prompts(args.prompt, args.prompts_file, args.prompt_limit)
    gpus = resolve_gpus(args.gpus)
    root = Path(args.output_dir) / datetime.utcnow().strftime("showo_ascr_parallel-%Y%m%d-%H%M%S")
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args, root, prompts, base_config)
    manifest = {
        "config": args.config,
        "output_root": str(root),
        "gpus": gpus,
        "max_workers": args.max_workers,
        "repeat_count": max(1, int(args.repeat_count)),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    manifest_path = root / "parallel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "manifest_path": str(manifest_path), "task_count": len(tasks), "gpus": gpus}, indent=2, sort_keys=True))
        return 0
    results = run_tasks(tasks, gpus, args.max_workers, logs_dir, project_root)
    suite = build_suite(results)
    suite["parallel"] = {"gpus": gpus, "max_workers": args.max_workers, "repeat_count": max(1, int(args.repeat_count)), "manifest_path": str(manifest_path)}
    suite_path = root / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    suite_md_path = root / "suite.md"
    suite_md_path.write_text(suite_to_markdown(suite), encoding="utf-8")
    print(json.dumps({"suite_path": str(suite_path), "suite_markdown_path": str(suite_md_path), "prompt_count": suite["prompt_count"], "total_ascr_insertions": suite["total_ascr_insertions"], "gpus": gpus}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

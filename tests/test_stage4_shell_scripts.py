import os
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_bash(command, env=None):
    if not shutil.which("bash"):
        raise unittest.SkipTest("bash is not available")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=str(ROOT),
        env=merged,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class Stage4ShellScriptTests(unittest.TestCase):
    def test_recovery_submit_dry_run_prints_expected_sbatch(self):
        completed = run_bash(
            "env DRY_RUN=1 MODE=submit CONFIG=cfg.yaml "
            "DATA_JSONL=data.jsonl OUTPUT_DIR=outdir "
            "bash scripts/training/run_stage4_recovery_submit.sh"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--gres=gpu:8", completed.stdout)
        self.assertIn("--export=ALL,CONFIG=cfg.yaml,NPROC=8", completed.stdout)
        self.assertIn("CHECKPOINT_EVERY_EPOCHS=1", completed.stdout)
        self.assertIn("EARLY_STOPPING_PATIENCE=3", completed.stdout)
        self.assertIn("EARLY_STOPPING_MIN_DELTA=0.0", completed.stdout)
        self.assertIn("DATA_JSONL=data.jsonl", completed.stdout)
        self.assertIn("OUTPUT_DIR=outdir", completed.stdout)

    def test_recovery_submit_handles_missing_slurm_state(self):
        completed = run_bash(
            "env MODE=recover JOB_ID=missing CURRENT_GPUS=8 "
            "PATH=/usr/bin:/bin bash scripts/training/run_stage4_recovery_submit.sh"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("No Slurm state found", completed.stderr)

    def test_hard256_eval_recovery_passes_chunk_env_to_sbatch(self):
        mockbin = ROOT / ".shell_test_bin"
        args_path = ROOT / ".shell_test_sbatch_args"
        shutil.rmtree(mockbin, ignore_errors=True)
        if args_path.exists():
            args_path.unlink()
        mockbin.mkdir()
        try:
            sbatch = mockbin / "sbatch"
            sbatch.write_bytes(
                b"#!/usr/bin/env bash\n"
                b"printf '%s\\n' \"$@\" > .shell_test_sbatch_args\n"
                b"echo 12345\n"
            )
            sbatch.chmod(0o755)
            completed = run_bash(
                "PATH=.shell_test_bin:/usr/bin:/bin MODE=submit_eval_recovery "
                "GRIDS=4 CHUNKS_PER_GPU=2 bash scripts/training/run_hard256_full_pipeline.sh"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            args = args_path.read_text(encoding="utf-8").splitlines()
        finally:
            shutil.rmtree(mockbin, ignore_errors=True)
            if args_path.exists():
                args_path.unlink()
        self.assertIn("--export=ALL,CONFIG,OUTPUT_ROOT,SAMPLES_PER_GPU,CHUNKS_PER_GPU", args)
        self.assertIn("jobs/stage4/stage4_multi_gpu_eval.sbatch", args)


if __name__ == "__main__":
    unittest.main()

# Reproducibility And Git Safety

## Local Validation

Run these before committing cleanup or experiment orchestration changes:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
python -m ascr.cli.run_stage1 --help
python -m ascr.cli.run_stage1 --dry-run --max-iterations 1 --output-dir outputs/local_smoke
python -m ascr.cli.preflight --mode local --config configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml --scan-secrets
git diff --check
```

The dry-run output goes under ignored `outputs/`; remove it after inspection if
you want a completely clean local tree.

## Secret Policy

Never commit real API keys, tokens, credential files, `.env` files, model access
tokens, or private server paths that encode credentials. Use `.env.template` as
documentation only. If a key has ever appeared in a committed file, rotate or
revoke it before pushing again and decide whether Git history needs to be purged.

External API judges read `OFOX_API_KEY` from the environment:

```bash
export OFOX_API_KEY='<your-ofox-api-key>'
```

Do not write the value into a script, Slurm file, notebook, or README.

## Safe Git Workflow

Use manual staging and a review branch:

```bash
git switch -c codex/server-ready-cleanup
python -m unittest discover -s tests
python -m ascr.cli.preflight --mode local --config configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml --scan-secrets
git diff --check
git status --short
git add <intentional files only>
git commit -m "Prepare ASCR for reproducible server runs"
git push -u origin codex/server-ready-cleanup
```

The maintenance helper `scripts/maintenance/sync_github.sh` is validation-only
and intentionally does not stage, commit, or push.

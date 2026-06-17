# Git Sync Guide

Use this workflow before pushing ASCR changes.

## Validate

```bash
git status --short --branch
git diff --stat
python scripts/smoke_test.py
python -m ascr.cli.preflight --mode local \
  --config configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml \
  --scan-secrets
git diff --check
```

## Check For Unsafe Files

```bash
rg -n -i "sk-|api_key|token|secret|password|OPENAI_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|GOOGLE_API_KEY|ANTHROPIC_API_KEY" \
  -g "!docs/archive/**" -g "!.git/**" .
git ls-files | rg --pcre2 "(^models/|^outputs/|^logs/|^checkpoints/|\\.safetensors$|\\.bin$|\\.pt$|\\.pth$|\\.ckpt$|\\.npy$|\\.npz$)"
```

Review matches manually. Words such as `token` are expected in ASCR source; real
secret values, model weights, checkpoints, generated outputs, caches, and private
datasets must not be committed.

## Commit And Push

```bash
git add <intentional project files only>
git status --short
git commit -m "Clear message describing the change"
git push
```

If the branch has no upstream:

```bash
git push -u origin HEAD
```

Never force-push and never rewrite history during normal experiment sync.

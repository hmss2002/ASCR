# Local Windows Setup

Use a project-local virtual environment for ASCR. Do not install ASCR dependencies
into Anaconda `base` unless you intentionally want all projects to share them.

## One-Time Setup

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/bootstrap_local.ps1
```

This creates `.venv`, upgrades `pip`, and installs ASCR with lightweight
development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

The `.venv` directory is ignored by Git and should never be committed.

## Activate Later

Whenever you reopen this project:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/activate_local.ps1
```

Or activate directly:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
python scripts/smoke_test.py
```

## What To Install Locally

For this Windows laptop, install only lightweight development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Avoid installing CUDA PyTorch, Lumina, Qwen, MMaDA, or model weights locally
unless you specifically want to debug heavy inference on this laptop. Heavy model
inference should happen on the Linux GPU server.

## Notes For Codex

When working in this repository locally, prefer:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

If `.venv` is missing, create it with `scripts/setup/bootstrap_local.ps1`.

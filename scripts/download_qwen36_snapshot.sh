#!/usr/bin/env bash
set -euo pipefail

cd /grp01/cds_bdai/JianyuZhang/ASCR
source .venv/bin/activate

export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}
export LOCAL_DIR=${LOCAL_DIR:-models/qwen3.6-35b-a3b}
export HF_DOWNLOAD_WORKERS=${HF_DOWNLOAD_WORKERS:-8}

mkdir -p "$(dirname "$LOCAL_DIR")"

python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ['MODEL_ID'], local_dir=os.environ['LOCAL_DIR'], local_dir_use_symlinks=False, max_workers=int(os.environ.get('HF_DOWNLOAD_WORKERS', '8')))"
python -c "import os; from pathlib import Path; root=Path(os.environ['LOCAL_DIR']); shards=sorted(root.glob('*.safetensors')); total=sum(path.stat().st_size for path in root.rglob('*') if path.is_file()); print(f'Qwen3.6 snapshot ready: {root} files={sum(1 for path in root.rglob('*') if path.is_file())} safetensors={len(shards)} size_gib={total / 1024**3:.2f}')"

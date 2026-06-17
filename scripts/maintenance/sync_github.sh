#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
cd "${PROJECT_ROOT}"

cat <<'EOF'
This helper is validation-only. It no longer stages, commits, or pushes files.

Safe sync workflow:
  git switch -c codex/server-ready-cleanup
  python -m unittest discover -s tests
  python -m ascr.cli.preflight --mode local --config configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml --scan-secrets
  git diff --check
  git status --short
  git add <intentional files only>
  git commit -m "Prepare ASCR for reproducible server runs"
  git push -u origin codex/server-ready-cleanup
EOF

git status --short
python -m unittest discover -s tests
python -m ascr.cli.preflight --mode local --config configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml --scan-secrets
git diff --check

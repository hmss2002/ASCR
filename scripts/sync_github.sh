#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"
message=${1:-"Update ASCR project"}
git status --short
git add README.md .gitignore pyproject.toml requirements configs ascr scripts jobs docs tests data
git commit -m "${message}"
git push

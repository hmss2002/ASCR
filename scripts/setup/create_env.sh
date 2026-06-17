#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)
cd "${PROJECT_ROOT}"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
if python -m pip install -e . -r requirements/base.txt; then
  printf "Editable ASCR install completed.\n"
else
  printf "Editable install failed; continuing with project-root python -m workflow.\n"
fi

printf "ASCR environment ready at %s/.venv\n" "${PROJECT_ROOT}"
printf "Activate with: source %s/.venv/bin/activate\n" "${PROJECT_ROOT}"

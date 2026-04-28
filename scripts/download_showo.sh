#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "${PROJECT_ROOT}"

SHOWO_REPO_URL=${SHOWO_REPO_URL:-https://github.com/showlab/Show-o.git}
SHOWO_REPO_DIR=${SHOWO_REPO_DIR:-external/Show-o}
SHOWO_MODEL_ROOT=${SHOWO_MODEL_ROOT:-models}
SHOWO_REVISION=${SHOWO_REVISION:-main}
SKIP_WEIGHTS=${SKIP_WEIGHTS:-0}
UPDATE_SHOWO_SOURCE=${UPDATE_SHOWO_SOURCE:-0}

mkdir -p "$(dirname "${SHOWO_REPO_DIR}")" "${SHOWO_MODEL_ROOT}"

if [ -d "${SHOWO_REPO_DIR}/.git" ]; then
  printf "Show-o source already exists at %s\n" "${SHOWO_REPO_DIR}"
  if [ "${UPDATE_SHOWO_SOURCE}" = "1" ]; then
    git -C "${SHOWO_REPO_DIR}" fetch --depth 1 origin "${SHOWO_REVISION}"
    git -C "${SHOWO_REPO_DIR}" checkout -q FETCH_HEAD
  else
    printf "UPDATE_SHOWO_SOURCE=0; reusing existing source checkout.\n"
  fi
else
  git clone --depth 1 --branch "${SHOWO_REVISION}" "${SHOWO_REPO_URL}" "${SHOWO_REPO_DIR}"
fi

if [ "${SKIP_WEIGHTS}" = "1" ]; then
  printf "SKIP_WEIGHTS=1; source checkout prepared only.\n"
  exit 0
fi

if [ -d .venv ]; then
  source .venv/bin/activate
fi

export HF_HOME=${HF_HOME:-${PROJECT_ROOT}/.hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export HF_HUB_DISABLE_SYMLINKS_WARNING=${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

python -m pip install -r requirements/showo_download.txt
SHOWO_MODEL_ROOT="${SHOWO_MODEL_ROOT}" python scripts/download_showo_models.py

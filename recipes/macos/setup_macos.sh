#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi

  local candidate
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew was not found. Install it first: https://brew.sh/"
  exit 1
fi

PYTHON_CMD="$(find_python || true)"
if [[ -z "${PYTHON_CMD}" ]]; then
  echo "No suitable Python interpreter was found."
  echo "Install Python 3.12 or 3.13 with Homebrew, for example:"
  echo "  brew install python@3.13"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for m4b packaging and reliable audio merging."
  echo "Install it with:"
  echo "  brew install ffmpeg"
  exit 1
fi

cd "${REPO_DIR}"

echo "Using Python interpreter: ${PYTHON_CMD}"
"${PYTHON_CMD}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cat <<'EOF'

macOS setup complete.

Quick smoke test:
  bash scripts/run_preview_macos.sh

OpenAI-compatible run:
  See docs/macos.md
EOF

#!/usr/bin/env bash
# Install day script for macOS / Linux. Run from the repo root.
set -euo pipefail

if ! command -v ollama >/dev/null; then
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! command -v uv >/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --all-extras
if command -v npm >/dev/null; then
  (cd frontend && npm install && npm run build)
else
  echo "npm not found; frontend not built. Install Node 22 and run: cd frontend && npm install && npm run build"
fi

uv run python scripts/pull_models.py
mkdir -p "${LOA_DATA_DIR:-$HOME/LocalOfficeAI}"
echo
echo "Done. Start with: desktop/launcher/start.sh"

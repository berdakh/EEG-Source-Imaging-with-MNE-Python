#!/usr/bin/env bash
# Starts Ollama if needed, then the backend, then opens the browser.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Starting Ollama..."
  (ollama serve >/dev/null 2>&1 &)
  sleep 2
fi

PORT="${LOA_PORT:-8000}"
echo "Local Office AI on http://127.0.0.1:$PORT"
( sleep 2; (open "http://127.0.0.1:$PORT" 2>/dev/null || xdg-open "http://127.0.0.1:$PORT" 2>/dev/null) ) &
exec uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "$PORT"

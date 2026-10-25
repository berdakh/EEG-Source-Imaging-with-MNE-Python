@echo off
REM Starts Ollama if needed, then the backend, then opens the browser.
cd /d "%~dp0\..\.."
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1 || (start "" /B ollama serve && timeout /t 2 >nul)
if "%LOA_PORT%"=="" set LOA_PORT=8000
echo Local Office AI on http://127.0.0.1:%LOA_PORT%
start "" http://127.0.0.1:%LOA_PORT%
uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port %LOA_PORT%

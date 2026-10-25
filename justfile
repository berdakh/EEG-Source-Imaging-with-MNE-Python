# Local Office AI developer commands. Install just: https://github.com/casey/just

set shell := ["bash", "-cu"]

default:
    @just --list

# Install Python and frontend dependencies
setup:
    uv sync --all-extras
    cd frontend && npm install

# Run backend (reload) and frontend dev server together
dev:
    (uv run uvicorn app.main:app --app-dir backend --reload --port 8000 &) ; cd frontend && npm run dev

# Build the frontend into frontend/dist
build:
    cd frontend && npm run build

# Run the backend serving the built UI
run:
    uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# Lint and test
check:
    uv run ruff check .
    uv run pytest -q

# Pull the models for this machine's hardware tier
models:
    uv run python scripts/pull_models.py

# Verify no outbound connections while exercising the app
offline:
    uv run python scripts/offline_check.py

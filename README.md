# Local Office AI

A free, fully offline AI assistant for small offices. Nothing leaves the
machine after install. The first vertical is a church office pilot; later
verticals (law, CPA, HOA management) are added as **packs** of prompts,
templates and YAML workflows, not as application code.

**Working name.** Rename the repo, the `LOA_` env prefix and the data folder
once the LLC has a brand.

## What it does (church pack)

| Feature | What the user does | What comes out |
|---|---|---|
| Sermon Studio | Drops in Sunday's recording | Transcript, summary, key points, discussion guide, blog draft, social posts, bulletin blurb |
| Announcement Hub | Pastes the week's messy announcements | Bulletin blurb, newsletter section, email, social post, slide text |
| Ask the Office | Asks a question | Answer with file and page citations from the church's own documents |
| Letter Drafter | Picks a template, fills fields | A Word letter on the church letterhead |

Principles: local only, human approves everything, citations or nothing,
workflows over chat, vertical content is configuration.

## Stack

Python 3.11+ / FastAPI backend, SQLite, BM25 + local embeddings for hybrid
search, [Ollama](https://ollama.com) for inference and embeddings,
faster-whisper for transcription, React + Vite frontend served by the backend.

## Quick start (development)

```bash
# 1. Install Ollama and pull models
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 2. Backend
uv sync --all-extras
uv run uvicorn app.main:app --app-dir backend --reload --port 8000

# 3. Frontend (separate terminal)
cd frontend && npm install && npm run dev      # http://localhost:5173, proxies /api to :8000
```

Or with [just](https://github.com/casey/just): `just setup`, `just dev`.

Production-style run: `just build` then `just run` serves the built UI from the
backend at http://localhost:8000.

## Configuration

Environment variables (prefix `LOA_`), or a `.env` file. See `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `LOA_DATA_DIR` | `~/LocalOfficeAI` | Database, uploaded files, outputs, logs. Backup = copy this folder. |
| `LOA_PACK` | `church` | Which pack in `packs/` to load |
| `LOA_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LOA_CHAT_MODEL` | `qwen3:8b` | Generation model |
| `LOA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `LOA_WHISPER_MODEL` | `small` | faster-whisper model size |
| `LOA_PIN` | empty | Office PIN; empty disables login |

## Layout

```
backend/app/      FastAPI app: db, llm, ingest, rag, transcribe, workflows, render, jobs, audit, api
frontend/         React + Vite UI
packs/church/     prompts, workflows, templates, sample data for the church vertical
desktop/          launcher scripts now, Tauri shell later
scripts/          install, model pull, offline check, synthetic data
docs/             scope, architecture, pilot runbook, discovery notes
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

Tests never require Ollama; the LLM is stubbed.

## License

Not yet decided. All rights reserved until a license file is added.

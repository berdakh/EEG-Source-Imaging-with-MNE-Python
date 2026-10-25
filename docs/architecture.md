# Architecture

## One sentence
A FastAPI backend owns a SQLite database and talks to a local Ollama server;
packs of YAML workflows and Jinja prompts describe what the app does; a React
UI calls the backend's JSON API.

## Request flow for "Ask the Office"
1. UI posts the question to `/api/ask`.
2. The `ask` workflow's `retrieve` step runs hybrid search: BM25 over chunk
   text plus cosine similarity over stored embeddings, fused by reciprocal rank.
3. The `generate` step renders `prompts/ask.md` with the hits and the question
   and sends it to Ollama.
4. The answer and the hits (title, page, text) come back; the UI shows the
   answer with numbered citations and the passages beside it.
5. The audit log records the question and the chunk ids that were retrieved.

## Packs
`packs/<name>/pack.yaml` declares navigation and hardware tiers. Each
`workflows/*.yaml` is a list of steps sharing one state dict keyed by input
names and step ids. Step types: `generate`, `retrieve`, `transcribe`,
`extract`, `render`. Prompts are Jinja templates with access to the whole
state. Adding a vertical means adding a pack, not touching `backend/`.

## Data
Everything lives in one data directory (`LOA_DATA_DIR`): `app.db`, `files/`
(originals), `outputs/` (generated .md and .docx), `logs/`. Backup is a copy
of that folder. Models are managed by Ollama.

## Long tasks
Transcription and embedding run through `app.jobs.JobQueue`, an in-process
asyncio task per job with status and progress persisted in the `jobs` table.
The UI polls `/api/jobs/{id}`.

## What is deliberately missing
- Authentication beyond a single office PIN (not yet wired).
- A vector database. Embeddings are float32 blobs searched in numpy; move to
  sqlite-vec when a library passes a few hundred thousand chunks.
- A desktop shell. Phase 1 is a launcher script and a browser tab.

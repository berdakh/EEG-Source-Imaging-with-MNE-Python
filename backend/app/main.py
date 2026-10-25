"""FastAPI entry point. Serves /api and, when built, the frontend from frontend/dist."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.config import Settings, get_settings
from app.db import connect, init_db
from app.jobs import JobQueue
from app.llm import OllamaClient
from app.workflows import load_pack


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s.ensure_dirs()
        conn = connect(s.db_path)
        init_db(conn)
        app.state.settings = s
        app.state.conn = conn
        app.state.pack = load_pack(s.packs_dir, s.pack)
        app.state.llm = OllamaClient(s.ollama_url, s.chat_model, s.embed_model)
        app.state.jobs = JobQueue(conn)
        try:
            yield
        finally:
            await app.state.llm.aclose()
            conn.close()

    app = FastAPI(title="Local Office AI", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)

    dist = s.frontend_dist
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            candidate = dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        async def root():
            return JSONResponse(
                {
                    "message": "Local Office AI backend is running. Build the frontend "
                    "(cd frontend && npm run build) or open /docs."
                }
            )

    return app


app = create_app()

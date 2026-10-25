"""Document library: upload, list, delete. Embedding runs as a background job."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.audit import record
from app.config import Settings
from app.ingest import load_file
from app.ingest.loaders import SUPPORTED
from app.jobs import JobQueue
from app.llm import OllamaClient
from app.rag import add_document, embed_document, list_documents, remove_document

from .deps import get_conn, get_jobs, get_llm, get_settings_dep

router = APIRouter(prefix="/library", tags=["library"])


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name) or "upload"


@router.get("")
async def documents(conn=Depends(get_conn)):
    return list_documents(conn)


@router.post("")
async def upload(
    file: UploadFile,
    conn=Depends(get_conn),
    s: Settings = Depends(get_settings_dep),
    llm: OllamaClient = Depends(get_llm),
    jobs: JobQueue = Depends(get_jobs),
):
    name = _safe_name(file.filename or "upload")
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type {ext}. Allowed: {sorted(SUPPORTED)}")
    dest = s.files_dir / name
    dest.write_bytes(await file.read())
    pages = load_file(dest)
    doc_id = add_document(
        conn, Path(name).stem, str(dest), ext.lstrip("."), pages, s.chunk_size, s.chunk_overlap
    )
    record(conn, "office", "library.upload", {"document_id": doc_id, "name": name})

    async def job(progress):
        n = await embed_document(conn, doc_id, llm.embed, s.embed_model)
        return {"document_id": doc_id, "chunks_embedded": n}

    job_id = jobs.submit("embed", job)
    return {"document_id": doc_id, "job_id": job_id, "pages": len(pages)}


@router.delete("/{doc_id}")
async def delete(doc_id: int, conn=Depends(get_conn)):
    remove_document(conn, doc_id)
    record(conn, "office", "library.delete", {"document_id": doc_id})
    return {"ok": True}

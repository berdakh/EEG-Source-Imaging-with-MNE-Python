"""Documents and chunks in SQLite, embeddings as float32 blobs."""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable

import numpy as np

from app.ingest import Page, chunk_text

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def add_document(
    conn: sqlite3.Connection,
    title: str,
    path: str,
    kind: str,
    pages: list[Page],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    cur = conn.execute(
        "INSERT INTO documents (title, path, kind) VALUES (?, ?, ?)", (title, path, kind)
    )
    doc_id = int(cur.lastrowid)
    idx = 0
    for page in pages:
        for piece in chunk_text(page.text, chunk_size, chunk_overlap):
            conn.execute(
                "INSERT INTO chunks (document_id, idx, page, text) VALUES (?, ?, ?, ?)",
                (doc_id, idx, page.page, piece),
            )
            idx += 1
    conn.commit()
    return doc_id


async def embed_document(
    conn: sqlite3.Connection, doc_id: int, embed: EmbedFn, model: str, batch: int = 32
) -> int:
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE document_id = ? ORDER BY idx", (doc_id,)
    ).fetchall()
    count = 0
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        vectors = await embed([r["text"] for r in part])
        for row, vec in zip(part, vectors, strict=True):
            arr = np.asarray(vec, dtype=np.float32)
            conn.execute(
                "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, model, dim, embedding) "
                "VALUES (?, ?, ?, ?)",
                (row["id"], model, arr.size, arr.tobytes()),
            )
            count += 1
    conn.commit()
    return count


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT d.id, d.title, d.kind, d.created_at, COUNT(c.id) AS chunks "
        "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
        "GROUP BY d.id ORDER BY d.created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def remove_document(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()

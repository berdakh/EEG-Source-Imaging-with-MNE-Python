"""In-process background jobs for long tasks (transcription, embedding). State lives in SQLite."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import traceback
import uuid
from collections.abc import Awaitable, Callable

ProgressFn = Callable[[float], None]
JobFn = Callable[[ProgressFn], Awaitable[dict]]


class JobQueue:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._tasks: dict[str, asyncio.Task] = {}

    def submit(self, kind: str, fn: JobFn) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO jobs (id, kind, status) VALUES (?, ?, 'queued')", (job_id, kind)
        )
        self.conn.commit()
        self._tasks[job_id] = asyncio.create_task(self._run(job_id, fn))
        return job_id

    def _set(self, job_id: str, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE jobs SET {cols}, updated_at = datetime('now') WHERE id = ?",
            (*fields.values(), job_id),
        )
        self.conn.commit()

    async def _run(self, job_id: str, fn: JobFn) -> None:
        self._set(job_id, status="running")

        def progress(p: float) -> None:
            self._set(job_id, progress=max(0.0, min(1.0, p)))

        try:
            result = await fn(progress)
            self._set(job_id, status="done", progress=1.0, result=json.dumps(result))
        except Exception as e:  # noqa: BLE001 - job failures are recorded, not raised
            self._set(job_id, status="error", error=f"{e}\n{traceback.format_exc()}")
        finally:
            self._tasks.pop(job_id, None)

    def get(self, job_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d

    def list(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, kind, status, progress, created_at, updated_at FROM jobs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

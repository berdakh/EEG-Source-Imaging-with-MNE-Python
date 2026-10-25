"""Activity log: who did what, when, and with which documents."""

from __future__ import annotations

import json
import sqlite3


def record(conn: sqlite3.Connection, actor: str, action: str, detail: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_log (actor, action, detail) VALUES (?, ?, ?)",
        (actor, action, json.dumps(detail or {}, ensure_ascii=False)),
    )
    conn.commit()


def recent(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    rows = conn.execute(
        "SELECT id, ts, actor, action, detail FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["detail"] = json.loads(d["detail"] or "{}")
        out.append(d)
    return out

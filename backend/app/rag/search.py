"""Hybrid retrieval: BM25 over chunk text plus cosine over embeddings, fused with RRF."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25L

from .store import EmbedFn

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    title: str
    page: int | None
    text: str
    score: float


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


async def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    embed: EmbedFn | None,
    k: int = 6,
    candidates: int = 25,
) -> list[Hit]:
    rows = conn.execute(
        "SELECT c.id, c.document_id, c.page, c.text, d.title "
        "FROM chunks c JOIN documents d ON d.id = c.document_id"
    ).fetchall()
    if not rows:
        return []
    by_id = {r["id"]: r for r in rows}

    # Lexical ranking
    # BM25L's idf is always positive (Okapi's goes to zero or negative on tiny corpora),
    # so it orders sensibly even with two documents. Candidacy still requires that the
    # chunk share at least one token with the query, since BM25L scores every chunk.
    doc_tokens = [_tokenize(r["text"]) for r in rows]
    q_tokens = _tokenize(query)
    q_set = set(q_tokens)
    lex_scores = BM25L(doc_tokens).get_scores(q_tokens)
    order = sorted(
        (i for i, t in enumerate(doc_tokens) if q_set & set(t)),
        key=lambda i: -lex_scores[i],
    )
    lex_rank = [rows[i]["id"] for i in order[:candidates]]

    rankings = [lex_rank]

    # Semantic ranking, only if we have an embedder and stored vectors
    if embed is not None:
        emb_rows = conn.execute("SELECT chunk_id, embedding, dim FROM chunk_embeddings").fetchall()
        if emb_rows:
            qvec = np.asarray((await embed([query]))[0], dtype=np.float32)
            mat = np.stack(
                [np.frombuffer(r["embedding"], dtype=np.float32, count=r["dim"]) for r in emb_rows]
            )
            norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(qvec) or 1.0)
            sims = mat @ qvec / np.where(norms == 0, 1.0, norms)
            order = np.argsort(-sims)[:candidates]
            rankings.append([emb_rows[i]["chunk_id"] for i in order])

    fused = _rrf(rankings)
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    hits = []
    for cid, score in top:
        r = by_id.get(cid)
        if r is None:
            continue
        hits.append(
            Hit(
                chunk_id=cid,
                document_id=r["document_id"],
                title=r["title"],
                page=r["page"],
                text=r["text"],
                score=score,
            )
        )
    return hits

"""Shared fixtures. Tests never talk to Ollama; the LLM is a stub."""

from __future__ import annotations

import re
import sqlite3
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import connect, init_db
from app.main import create_app
from app.workflows import load_pack

REPO_ROOT = Path(__file__).resolve().parents[2]


class StubLLM:
    """Echoes the prompt head so tests can assert the template was rendered."""

    def __init__(self):
        self.calls: list[dict] = []

    async def generate(self, prompt, system=None, model=None, options=None):
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        return "STUB: " + prompt[:60].replace("\n", " ")

    async def embed(self, texts, model=None):
        # Deterministic hashed bag-of-words, so overlapping vocabulary means higher cosine.
        dim = 64
        out = []
        for t in texts:
            vec = [0.0] * dim
            for tok in re.findall(r"\w+", t.lower()):
                vec[zlib.crc32(tok.encode()) % dim] += 1.0
            out.append(vec)
        return out

    async def is_available(self):
        return False

    async def list_models(self):
        return []

    async def aclose(self):
        return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        packs_dir=REPO_ROOT / "packs",
        frontend_dist=tmp_path / "no-dist",
        ollama_url="http://127.0.0.1:1",  # nothing listens here
    )


@pytest.fixture
def conn(settings: Settings) -> sqlite3.Connection:
    settings.ensure_dirs()
    c = connect(settings.db_path)
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def pack(settings: Settings):
    return load_pack(settings.packs_dir, settings.pack)


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def client(settings: Settings, stub_llm: StubLLM):
    app = create_app(settings)
    with TestClient(app) as c:
        app.state.llm = stub_llm
        yield c

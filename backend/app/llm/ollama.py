"""Thin async client for the Ollama HTTP API (chat, embeddings, model listing)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embed_model: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def is_available(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        try:
            r = await self._client.get("/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except httpx.HTTPError:
            return []

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": True,
            "options": options or {},
        }
        async with self._client.stream("POST", "/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        options: dict | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        parts = [t async for t in self.chat_stream(messages, model=model, options=options)]
        return "".join(parts).strip()

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        r = await self._client.post(
            "/api/embed", json={"model": model or self.embed_model, "input": texts}
        )
        r.raise_for_status()
        return r.json()["embeddings"]

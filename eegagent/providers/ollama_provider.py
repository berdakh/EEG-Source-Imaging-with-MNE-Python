"""Local open-weight models through Ollama's chat API with native tool calling."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .base import Provider, ToolCall, Turn, new_call_id

DEFAULT_MODEL = "qwen3:8b"


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(
        self, model: str | None = None, host: str | None = None, timeout: float = 600.0, num_ctx: int = 16384
    ):
        self.model = model or os.environ.get("EEGAGENT_MODEL") or DEFAULT_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("text", "")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.args}} for c in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                for r in m["results"]:
                    out.append({"role": "tool", "content": r.content, "tool_name": r.name})
        return out

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        body = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "tools": self._tools(tools),
            "stream": False,
            "options": {"num_ctx": self.num_ctx},
        }
        try:
            r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            return Turn(
                text=f"Ollama error {e.response.status_code}: {e.response.text[:300]}", stop_reason="error"
            )
        except httpx.HTTPError as e:
            return Turn(
                text=f"Cannot reach Ollama at {self.host}: {e}. Is `ollama serve` running?",
                stop_reason="error",
            )
        data = r.json()
        msg = data.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(id=tc.get("id") or new_call_id(), name=fn.get("name", ""), args=args))
        usage = {
            "input_tokens": int(data.get("prompt_eval_count", 0)),
            "output_tokens": int(data.get("eval_count", 0)),
        }
        stop = "tool_use" if calls else ("max_tokens" if data.get("done_reason") == "length" else "end_turn")
        return Turn(text=(msg.get("content") or "").strip(), tool_calls=calls, stop_reason=stop, usage=usage)

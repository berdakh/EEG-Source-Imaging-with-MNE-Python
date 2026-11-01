"""Claude via the official Anthropic SDK (Messages API, native tool use)."""

from __future__ import annotations

import json
from typing import Any

from .base import Provider, ToolCall, Turn

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 16000,
        effort: str | None = None,
        fallbacks: bool = True,
    ):
        import anthropic

        from ..config import setting

        api_key = setting("anthropic_api_key")  # None -> the SDK's own credential resolution
        self.client = anthropic.Anthropic(api_key=api_key or None)
        self.model = setting("model", model, DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        self._anthropic = anthropic

    # ---- format conversion ---------------------------------------------------
    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            out.append(
                {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            )
        return out

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                if m.get("raw") is not None:
                    out.append({"role": "assistant", "content": m["raw"]})
                else:
                    blocks: list[dict[str, Any]] = []
                    if m.get("text"):
                        blocks.append({"type": "text", "text": m["text"]})
                    for c in m.get("tool_calls", []):
                        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
                    out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in m["results"]
                        ],
                    }
                )
        return out

    # ---- call ----------------------------------------------------------------
    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=self._tools(tools),
            messages=self._messages(messages),
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            if self.fallbacks:
                # Server-side refusal fallback: if a safety classifier declines, the API re-runs the
                # request on a fallback model inside the same call.
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                resp = self.client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as e:
            return Turn(text=f"API error {e.status_code}: {e.message}", stop_reason="error")
        except self._anthropic.APIConnectionError as e:
            return Turn(text=f"Connection error: {e}", stop_reason="error")

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                calls.append(ToolCall(id=block.id, name=block.name, args=args))
        stop = resp.stop_reason or "end_turn"
        if stop == "refusal":
            text_parts.append("The model declined this request.")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        if getattr(resp.usage, "cache_read_input_tokens", None):
            usage["cache_read_input_tokens"] = resp.usage.cache_read_input_tokens
        # keep the native content (thinking blocks included) for replay
        raw = [b.model_dump(exclude_none=True) for b in resp.content]
        return Turn(
            text="\n".join(text_parts).strip(), tool_calls=calls, stop_reason=stop, usage=usage, raw=raw
        )

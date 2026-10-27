"""Provider factory: one line switches between Claude and a local Ollama model."""

from __future__ import annotations

import os

from .base import Provider, ToolCall, ToolResult, Turn
from .scripted import RuleProvider, ScriptedProvider

PROVIDERS = ("anthropic", "ollama", "rule")


def get_provider(name: str | None = None, model: str | None = None, **kwargs) -> Provider:
    name = (name or os.environ.get("EEGAGENT_PROVIDER") or "anthropic").lower()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model, **kwargs)
    if name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(model=model, **kwargs)
    if name == "rule":
        return RuleProvider()
    raise ValueError(f"Unknown provider '{name}'. Choose from {PROVIDERS}.")


__all__ = [
    "Provider",
    "ToolCall",
    "ToolResult",
    "Turn",
    "ScriptedProvider",
    "RuleProvider",
    "get_provider",
    "PROVIDERS",
]

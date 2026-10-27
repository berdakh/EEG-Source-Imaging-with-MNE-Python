"""Provider-neutral message format and the Provider interface.

Conversation history is a list of dicts:

    {"role": "user", "text": "..."}
    {"role": "assistant", "text": "...", "tool_calls": [ToolCall...], "raw": <provider-native content>}
    {"role": "tool", "results": [ToolResult...]}

``raw`` lets a provider replay its own assistant turn byte-for-byte (Anthropic needs its
thinking blocks preserved); other providers ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    """One assistant response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens | refusal | error
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class Provider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        """One model call. ``tools`` are provider-neutral specs from eegagent.tools.tool_specs()."""


def new_call_id(prefix: str = "call") -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:12]}"

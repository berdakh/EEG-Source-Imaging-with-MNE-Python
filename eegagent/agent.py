"""The agent loop: model proposes tool calls, we run them against the Session, repeat."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .providers.base import Provider, ToolResult, Turn
from .session import Session
from .tools import run_tool, to_json, tool_specs

SYSTEM_PROMPT = """You are EEG-Agent, an assistant that performs EEG source-imaging analyses with MNE-Python
through the tools provided. You work inside one analysis session whose state persists between tool calls.

Typical pipeline, in order:
1. load_recording  2. filter_data (and detect_bad_channels / set_average_reference if useful)
3. make_epochs  4. compute_evoked  5. compute_forward  6. compute_sources  7. cluster_test / generate_report.

Rules:
- Use the tools; never claim a step was done without calling it. Call describe_state if unsure what exists.
- If a tool returns an error, read it and fix the problem (usually a missing earlier step), then continue.
- Choose sensible defaults when the user does not specify parameters, and say which you used.
- Prefer the 'sphere' head model unless the user asks for fsaverage or the data has MRI information.
- Condition names default to event_<id>; use the ids reported by load_recording.
- When finished, give a concise summary: the steps taken with key parameters, the main numbers
  (peak channel/time, peak source location, significant clusters), and the report path if one was made.
- Do not invent results. Numbers in your summary must come from tool outputs."""


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    is_error: bool
    seconds: float


@dataclass
class AgentResult:
    text: str
    trace: list[ToolEvent] = field(default_factory=list)
    steps: int = 0
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [e.name for e in self.trace]


class Agent:
    def __init__(
        self,
        provider: Provider,
        session: Session | None = None,
        max_steps: int = 25,
        system_prompt: str = SYSTEM_PROMPT,
        on_event: Callable[[str, Any], None] | None = None,
    ):
        self.provider = provider
        self.session = session or Session()
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.on_event = on_event or (lambda kind, payload: None)
        self.history: list[dict[str, Any]] = []
        self.tools = tool_specs()

    def reset(self) -> None:
        self.history = []

    def run(self, user_message: str) -> AgentResult:
        """Handle one user turn; may involve many tool calls. History is kept for follow-ups."""
        self.history.append({"role": "user", "text": user_message})
        trace: list[ToolEvent] = []
        usage: dict[str, int] = {}
        turn = Turn()
        for step in range(1, self.max_steps + 1):
            turn = self.provider.complete(self.system_prompt, self.history, self.tools)
            for k, v in turn.usage.items():
                usage[k] = usage.get(k, 0) + int(v)
            self.history.append(
                {"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls, "raw": turn.raw}
            )
            if turn.text:
                self.on_event("text", turn.text)
            if turn.stop_reason == "error":
                return AgentResult(turn.text, trace, step, "error", usage)
            if not turn.tool_calls:
                return AgentResult(turn.text, trace, step, turn.stop_reason, usage)
            results = []
            for call in turn.tool_calls:
                self.on_event("tool_call", {"name": call.name, "args": call.args})
                t0 = time.perf_counter()
                result, is_error = run_tool(self.session, call.name, call.args)
                ev = ToolEvent(call.name, call.args, result, is_error, round(time.perf_counter() - t0, 3))
                trace.append(ev)
                self.on_event("tool_result", ev)
                results.append(
                    ToolResult(id=call.id, name=call.name, content=to_json(result), is_error=is_error)
                )
            self.history.append({"role": "tool", "results": results})
        # ran out of steps: ask the model for a summary without tools
        self.history.append(
            {"role": "user", "text": "You have reached the step limit. Summarise what was done so far."}
        )
        turn = self.provider.complete(self.system_prompt, self.history, [])
        self.history.append({"role": "assistant", "text": turn.text, "tool_calls": [], "raw": turn.raw})
        return AgentResult(turn.text, trace, self.max_steps, "max_steps", usage)

"""Deterministic providers for tests and the eval harness. No network, no model."""

from __future__ import annotations

from typing import Any

from .base import Provider, ToolCall, Turn, new_call_id


class ScriptedProvider(Provider):
    """Replays a fixed list of turns. Each item is either a string (final text) or a list of
    (tool_name, args) tuples (one assistant turn with those tool calls)."""

    name = "scripted"
    model = "scripted"

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def complete(self, system, messages, tools) -> Turn:
        self.calls.append({"n_messages": len(messages), "n_tools": len(tools)})
        if not self.script:
            return Turn(text="Done.", stop_reason="end_turn")
        item = self.script.pop(0)
        if isinstance(item, str):
            return Turn(text=item, stop_reason="end_turn")
        calls = [ToolCall(id=new_call_id(), name=n, args=a) for n, a in item]
        return Turn(text="", tool_calls=calls, stop_reason="tool_use")


class RuleProvider(Provider):
    """A tiny keyword planner that mimics what a competent model does, used to smoke-test the
    whole stack end to end without an LLM. Not clever; deliberately so."""

    name = "rule"
    model = "rule"

    def complete(self, system, messages, tools) -> Turn:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user = messages[last_user]["text"]
        # tools already called in this turn, plus everything from earlier turns (session state persists)
        done = [c.name for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])]
        this_turn = [
            c.name for m in messages[last_user:] if m["role"] == "assistant" for c in m.get("tool_calls", [])
        ]
        if "load_recording" not in done:
            done = this_turn
        text = user.lower()
        plan: list[tuple[str, dict]] = []
        if "load_recording" not in done:
            src = "sample" if "sample" in text else "simulated"
            for token in text.split():
                if token.endswith((".vhdr", ".fif", ".edf", ".set")):
                    src = token.strip("'\"`,")
            plan.append(("load_recording", {"source": src}))
        elif (
            any(k in text for k in ("filter", "band-pass", "bandpass", "notch", " hz"))
            and "filter_data" not in done
        ):
            plan.append(
                ("filter_data", {"l_freq": 1.0, "h_freq": 30.0, "notch": 50.0 if "notch" in text else None})
            )
        elif "bad channel" in text and "detect_bad_channels" not in done:
            plan.append(("detect_bad_channels", {}))
        elif (
            any(k in text for k in ("epoch", "source", "localis", "localiz", "cluster", "report", "evoked"))
            and "make_epochs" not in done
        ):
            plan.append(("make_epochs", {"tmin": -0.2, "tmax": 0.5}))
        elif "cluster" in text and "cluster_test" not in this_turn:
            plan.append(
                ("cluster_test", {"condition_a": "event_1", "condition_b": "event_2", "n_permutations": 100})
            )
        elif (
            any(k in text for k in ("source", "localis", "localiz", "report", "evoked"))
            and "compute_evoked" not in done
        ):
            plan.append(("compute_evoked", {"condition": "event_1"}))
        elif any(k in text for k in ("source", "localis", "localiz")) and "compute_forward" not in done:
            plan.append(("compute_forward", {"model": "sphere"}))
        elif any(k in text for k in ("source", "localis", "localiz")) and "compute_sources" not in done:
            plan.append(("compute_sources", {"method": "dSPM"}))
        elif "report" in text and "generate_report" not in this_turn:
            plan.append(("generate_report", {}))
        if not plan:
            return Turn(text=self._summary(messages), stop_reason="end_turn")
        return Turn(tool_calls=[ToolCall(new_call_id(), n, a) for n, a in plan], stop_reason="tool_use")

    @staticmethod
    def _summary(messages) -> str:
        """Summarise from the tool results in the history, the way a model is asked to."""
        import json

        latest: dict[str, dict] = {}
        for m in messages:
            if m["role"] == "tool":
                for r in m["results"]:
                    try:
                        latest[r.name] = json.loads(r.content)
                    except json.JSONDecodeError:
                        pass
        parts = [f"Steps run: {', '.join(latest)}."]
        if "compute_evoked" in latest and "peak_channel" in latest["compute_evoked"]:
            e = latest["compute_evoked"]
            parts.append(f"Sensor peak at {e['peak_channel']} ({round(e['peak_time_s'] * 1000)} ms).")
        if "compute_sources" in latest and "peak" in latest["compute_sources"]:
            pk = latest["compute_sources"]["peak"]
            parts.append(
                f"Peak source ({pk['method']}) at {pk['position_mm']} mm, t = {pk['time_s']:.3f} s "
                f"({round(pk['time_s'] * 1000)} ms)."
            )
        if "cluster_test" in latest and "n_significant" in latest["cluster_test"]:
            c = latest["cluster_test"]
            parts.append(f"Cluster test: {c['n_significant']} significant cluster(s) of {c['n_clusters']}.")
        if "generate_report" in latest and "path" in latest["generate_report"]:
            parts.append(f"Report: {latest['generate_report']['path']}")
        return " ".join(parts)

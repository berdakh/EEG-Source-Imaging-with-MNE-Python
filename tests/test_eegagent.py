"""Agent loop, tools and providers, all offline."""

import json

import pytest

from eegagent import Agent, Session
from eegagent.providers import RuleProvider, ScriptedProvider, get_provider
from eegagent.providers.anthropic_provider import AnthropicProvider
from eegagent.providers.base import ToolCall, ToolResult
from eegagent.providers.ollama_provider import OllamaProvider
from eegagent.tools import TOOLS, run_tool, tool_specs


@pytest.fixture
def session(tmp_path):
    return Session(workspace=tmp_path)


def test_tool_specs_are_valid_json_schema():
    specs = tool_specs()
    names = {s["name"] for s in specs}
    assert {"load_recording", "make_epochs", "compute_sources", "generate_report"} <= names
    for s in specs:
        schema = s["input_schema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        for r in schema["required"]:
            assert r in schema["properties"]
        json.dumps(s)


def test_tools_fail_gracefully_without_data(session):
    for name in ("filter_data", "make_epochs", "compute_sources", "generate_report"):
        result, is_error = run_tool(
            session, name, {"tmin": -0.1, "tmax": 0.1} if name == "make_epochs" else {}
        )
        assert is_error and "load_recording" in result["error"] or "first" in result["error"]
    result, is_error = run_tool(session, "nonexistent", {})
    assert is_error and "Unknown tool" in result["error"]
    result, is_error = run_tool(session, "filter_data", {"bogus": 1})
    assert is_error and "Bad arguments" in result["error"]


def test_full_tool_chain(session):
    r, e = run_tool(session, "load_recording", {"source": "simulated"})
    assert not e and r["event_ids"] == [1, 2]
    r, e = run_tool(session, "filter_data", {"l_freq": 1, "h_freq": 30, "notch": 50})
    assert not e
    r, e = run_tool(session, "detect_bad_channels", {})
    assert not e and isinstance(r["bad_channels"], list)
    r, e = run_tool(
        session, "make_epochs", {"tmin": -0.2, "tmax": 0.5, "event_names": {"strong": 1, "weak": 2}}
    )
    assert not e and set(r["conditions"]) == {"strong", "weak"}
    r, e = run_tool(session, "set_average_reference", {})
    assert not e and r["applied_to"] == "epochs"
    r, e = run_tool(session, "compute_evoked", {"condition": "strong"})
    assert not e and r["peak_channel"] in {"C3", "Cz", "FC1", "CP1"}
    r, e = run_tool(session, "compute_forward", {"model": "sphere", "spacing_mm": 20})
    assert not e and r["source_space"] == "volume"
    r, e = run_tool(session, "compute_sources", {"method": "dSPM", "peak_tmin": 0.05, "peak_tmax": 0.15})
    assert not e and r["peak"]["position_mm"][0] < 0
    r, e = run_tool(
        session, "cluster_test", {"condition_a": "strong", "condition_b": "weak", "n_permutations": 100}
    )
    assert not e and r["n_significant"] >= 1
    r, e = run_tool(session, "generate_report", {"title": "t"})
    assert not e and r["path"].endswith("report.html")
    state = session.describe()
    assert state["epochs"]["conditions"]["strong"] > 0 and "sources" in state and "cluster_test" in state
    assert len(session.log) >= 9


def test_agent_loop_with_scripted_provider(session):
    provider = ScriptedProvider(
        [
            [("load_recording", {"source": "simulated"})],
            [("make_epochs", {"tmin": -0.2, "tmax": 0.4}), ("describe_state", {})],  # parallel calls
            [("compute_sources", {})],  # error: no evoked / forward yet -> model should see is_error
            "Loaded, epoched; sources failed because no forward model.",
        ]
    )
    events = []
    agent = Agent(provider, session, on_event=lambda k, p: events.append(k))
    res = agent.run("load the simulated data and epoch it")
    assert res.tool_names == ["load_recording", "make_epochs", "describe_state", "compute_sources"]
    assert res.trace[-1].is_error and "compute_evoked" in res.trace[-1].result["error"]
    assert res.text.startswith("Loaded")
    assert res.steps == 4 and res.stop_reason == "end_turn"
    # history shape: user, assistant, tool, assistant, tool, assistant, tool, assistant
    roles = [m["role"] for m in agent.history]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"]
    assert len(agent.history[2]["results"]) == 1 and len(agent.history[4]["results"]) == 2
    assert "tool_call" in events and "tool_result" in events


def test_agent_step_limit(session):
    provider = ScriptedProvider([[("describe_state", {})]] * 10 + ["summary"])
    agent = Agent(provider, session, max_steps=3)
    res = agent.run("loop forever")
    assert res.stop_reason == "max_steps" and len(res.trace) == 3
    assert agent.history[-2]["text"].startswith("You have reached the step limit")


def test_rule_provider_end_to_end(session):
    agent = Agent(RuleProvider(), session)
    res = agent.run("Load the simulated recording, filter it, localise the sources and write a report")
    assert res.tool_names == [
        "load_recording",
        "filter_data",
        "make_epochs",
        "compute_evoked",
        "compute_forward",
        "compute_sources",
        "generate_report",
    ]
    assert all(not e.is_error for e in res.trace)
    assert session.results["report"]["path"].endswith("report.html")


def test_get_provider_names():
    assert isinstance(get_provider("rule"), RuleProvider)
    with pytest.raises(ValueError):
        get_provider("nope")


def test_anthropic_message_conversion():
    msgs = [
        {"role": "user", "text": "hi"},
        {
            "role": "assistant",
            "text": "calling",
            "tool_calls": [ToolCall("t1", "describe_state", {})],
            "raw": None,
        },
        {"role": "tool", "results": [ToolResult("t1", "describe_state", '{"a":1}', False)]},
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [],
            "raw": [{"type": "thinking", "thinking": "", "signature": "x"}, {"type": "text", "text": "done"}],
        },
    ]
    out = AnthropicProvider._messages(msgs)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "describe_state", "input": {}}
    assert out[2]["content"][0]["type"] == "tool_result" and out[2]["content"][0]["tool_use_id"] == "t1"
    assert out[3]["content"][0]["type"] == "thinking", "native content must be replayed verbatim"
    tools = AnthropicProvider._tools(tool_specs())
    assert tools[0]["input_schema"]["type"] == "object"


def test_ollama_message_conversion():
    msgs = [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("t1", "filter_data", {"l_freq": 1})]},
        {"role": "tool", "results": [ToolResult("t1", "filter_data", "{}", False)]},
    ]
    out = OllamaProvider._messages("sys", msgs)
    assert out[0]["role"] == "system"
    assert out[2]["tool_calls"][0]["function"] == {"name": "filter_data", "arguments": {"l_freq": 1}}
    assert out[3] == {"role": "tool", "content": "{}", "tool_name": "filter_data"}
    tools = OllamaProvider._tools(tool_specs())
    assert tools[0]["type"] == "function" and "parameters" in tools[0]["function"]


def test_ollama_unreachable_returns_error_turn():
    p = OllamaProvider(host="http://127.0.0.1:9", timeout=2)
    turn = p.complete("sys", [{"role": "user", "text": "x"}], [])
    assert turn.stop_reason == "error" and "Ollama" in turn.text


def test_every_tool_has_a_docstring_sized_description():
    for t in TOOLS.values():
        assert len(t.description) > 40, t.name

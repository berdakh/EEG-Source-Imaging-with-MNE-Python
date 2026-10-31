from eegagent.evals.runner import CHECKS, format_report, is_subsequence, load_cases, run_evals
from eegagent.providers import RuleProvider


def test_cases_reference_known_checks():
    cases = load_cases()
    assert len(cases) >= 6
    for c in cases:
        for chk in c.get("checks", []):
            assert chk in CHECKS, chk


def test_is_subsequence():
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])
    assert is_subsequence([], ["x"])


def test_rule_provider_eval_suite(tmp_path):
    summary = run_evals(RuleProvider(), workspace=tmp_path)
    by_id = {r["id"]: r for r in summary["cases"]}
    # the keyword planner handles every analysis case...
    for cid in (
        "load_and_epoch",
        "filter_then_epoch",
        "full_source_localisation",
        "cluster_statistics",
        "report_generation",
        "plan_missing_prerequisites",
    ):
        assert by_id[cid]["passed"], (cid, by_id[cid]["outcomes"], by_id[cid]["errors"])
    # ...but not the one that needs judgement, which proves the grader can fail a run
    assert not by_id["inspect_only"]["passed"]
    assert by_id["inspect_only"]["outcomes"]["no_forbidden_tools"] is False
    assert summary["n_passed"] == 6 and summary["results_path"].endswith(".json")
    assert "PASS" in format_report(summary) and "FAIL" in format_report(summary)


def test_peak_answer_checks_coordinates_and_unit_bearing_time(tmp_path):
    from eegagent.agent import AgentResult
    from eegagent.session import Session

    session = Session(workspace=tmp_path)
    session.results["sources"] = {"peak": {"position_mm": [-40.2, 0.1, 59.9], "time_s": 0.1}}
    check = CHECKS["answer_reports_peak"]
    for text in (
        "Peak source (dSPM) at [-40.2, 0.1, 59.9] mm, t = 0.100 s (100 ms).",
        "Peak source at (-40, 0, 60) mm and 100 milliseconds.",
        "Epoch starts at 200 ms. Peak source at (-40, 0, 60) mm and 100 ms.",
    ):
        assert check(session, AgentResult(text)), text
    for text in (
        "The epoch starts at 100 ms. The source peak is at (-40, 0, 60) mm at 200 ms.",
        "Peak at [-40, 0, 60] mm, t = 0.100 s (200 ms).",
        "Peak at [-40, 0, 60] mm. The epoch starts at 100 ms.",
        "Peak at [-40, 0, 60] mm\nThe epoch starts at 100 ms.",
        "Peak at 100 ms.",  # missing coordinates
        "Peak at [40, 0, 60] mm and 100 ms.",  # wrong hemisphere, right time
        "Peak at [-40, 0, 60] mm and 1100 ms.",  # substring collision
        "Peak at [-40, 0, 60] mm and 100.",  # no time unit
        "Peak at [-40, 0, 60] mm and 0.101 s.",  # outside rounding tolerance
        "Peak at [-40, 0, 60] cm and 100 ms.",  # wrong coordinate unit
        "Peak at [-40, 0, 60] mm, also [99, 99, 99] mm, at 100 ms.",
    ):
        assert not check(session, AgentResult(text)), text


def test_error_recovery_requires_actual_failure_then_success(tmp_path):
    from eegagent.agent import AgentResult, ToolEvent
    from eegagent.session import Session

    session = Session(workspace=tmp_path)
    check = CHECKS["recovered_error"]

    def event(name, error):
        return ToolEvent(name, {}, {}, error, 0.0)

    assert not check(session, AgentResult("", [event("compute_sources", False)]))
    assert not check(session, AgentResult("", [event("compute_sources", True)]))
    assert not check(
        session, AgentResult("", [event("compute_sources", False), event("compute_sources", True)])
    )
    assert not check(
        session, AgentResult("", [event("compute_sources", True), event("describe_state", False)])
    )
    assert check(session, AgentResult("", [event("compute_sources", True), event("compute_sources", False)]))
    assert not check(
        session,
        AgentResult(
            "", [event("compute_sources", True), event("compute_sources", False), event("make_epochs", True)]
        ),
    )


def test_scripted_provider_repairs_actual_missing_prerequisite(tmp_path):
    from eegagent import Agent, Session
    from eegagent.providers import ScriptedProvider

    session = Session(workspace=tmp_path)
    provider = ScriptedProvider(
        [
            [("compute_sources", {})],  # real tool failure, not a fabricated trace
            [("load_recording", {"source": "simulated"})],
            [("make_epochs", {"tmin": -0.2, "tmax": 0.5})],
            [("compute_evoked", {"condition": "event_1"})],
            [("compute_forward", {"model": "sphere"})],
            [("compute_sources", {"method": "dSPM"})],
            "Recovered after loading data and computing the missing prerequisites.",
        ]
    )
    result = Agent(provider, session).run("Compute sources, repairing missing prerequisites.")
    assert result.trace[0].is_error
    assert not result.trace[-1].is_error
    assert "sources" in session.describe()
    assert CHECKS["recovered_error"](session, result)
    # This proves the tool-loop contract, not that an untested LLM can recover.


def test_required_eval_case_cannot_be_hidden_by_aggregate_score(monkeypatch, capsys):
    from eegagent.cli import main

    summary = {"pass_rate": 0.9, "cases": [{"id": "inspect_only", "passed": False}], "report": "test"}
    monkeypatch.setattr("eegagent.evals.runner.run_evals", lambda *args, **kwargs: summary)
    assert (
        main(["--provider", "rule", "eval", "--min-pass-rate", "0.8", "--require-case", "inspect_only"]) == 1
    )
    summary["cases"][0]["passed"] = True
    assert (
        main(["--provider", "rule", "eval", "--min-pass-rate", "0.8", "--require-case", "inspect_only"]) == 0
    )
    assert main(["--provider", "rule", "eval", "--require-case", "missing_case"]) == 1
    capsys.readouterr()

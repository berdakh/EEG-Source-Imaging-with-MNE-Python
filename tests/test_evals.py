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
        "recover_from_missing_step",
    ):
        assert by_id[cid]["passed"], (cid, by_id[cid]["outcomes"], by_id[cid]["errors"])
    # ...but not the one that needs judgement, which proves the grader can fail a run
    assert not by_id["inspect_only"]["passed"]
    assert by_id["inspect_only"]["outcomes"]["no_forbidden_tools"] is False
    assert summary["n_passed"] == 6 and summary["results_path"].endswith(".json")
    assert "PASS" in format_report(summary) and "FAIL" in format_report(summary)

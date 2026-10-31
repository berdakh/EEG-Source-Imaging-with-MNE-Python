"""Eval runner: prompts in, tool traces and session state out, graded by deterministic checks.

Run with ``eegagent eval --provider anthropic`` (real model) or ``--provider rule``
(offline smoke test of the harness). Results are written as JSON under the workspace.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from ..agent import Agent, AgentResult
from ..providers.base import Provider
from ..session import Session

CASES_PATH = Path(__file__).with_name("cases.yaml")


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    with open(path or CASES_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(x == y for y in it) for x in needle)


# ---------------------------------------------------------------------------- named checks


def _check_epochs_window_matches(s: Session, r: AgentResult) -> bool:
    return s.epochs is not None and abs(s.epochs.tmin + 0.2) < 1e-6 and abs(s.epochs.tmax - 0.5) < 1e-2


def _check_filter_band_1_30(s: Session, r: AgentResult) -> bool:
    return (
        s.raw is not None
        and abs(s.raw.info["highpass"] - 1.0) < 0.01
        and abs(s.raw.info["lowpass"] - 30.0) < 0.01
    )


def _check_peak_in_left_hemisphere(s: Session, r: AgentResult) -> bool:
    peak = (s.results.get("sources") or {}).get("peak") or {}
    return bool(peak) and peak["position_mm"][0] < 0


def _check_peak_time_near_100ms(s: Session, r: AgentResult) -> bool:
    peak = (s.results.get("sources") or {}).get("peak") or {}
    return bool(peak) and 0.04 <= peak["time_s"] <= 0.16


def _check_answer_reports_peak(s: Session, r: AgentResult) -> bool:
    """The numbers in the answer must come from the tool output (no invented coordinates)."""
    peak = (s.results.get("sources") or {}).get("peak") or {}
    if not peak:
        return False
    t_ms = round(peak["time_s"] * 1000)
    text = r.text.replace(",", "")
    return str(t_ms) in text or f"{peak['time_s']:.2f}" in text or f"{peak['time_s']:.3f}" in text


def _check_has_significant_cluster(s: Session, r: AgentResult) -> bool:
    return (s.results.get("cluster_test") or {}).get("n_significant", 0) >= 1


def _check_report_file_exists(s: Session, r: AgentResult) -> bool:
    p = (s.results.get("report") or {}).get("path")
    return bool(p) and Path(p).exists() and Path(p).stat().st_size > 1000


def _check_no_unrecovered_errors(s: Session, r: AgentResult) -> bool:
    """Errors are allowed (that is the point of the case) but the last call of each errored tool must succeed."""
    last_status: dict[str, bool] = {}
    for ev in r.trace:
        last_status[ev.name] = ev.is_error
    return not any(last_status.values())


CHECKS = {name[len("_check_") :]: fn for name, fn in globals().items() if name.startswith("_check_")}


# ---------------------------------------------------------------------------- grading


def grade(case: dict[str, Any], session: Session, result: AgentResult) -> dict[str, Any]:
    outcomes: dict[str, bool] = {}
    names = result.tool_names
    if case.get("expect_tools"):
        outcomes["tools_in_order"] = is_subsequence(case["expect_tools"], names)
    if case.get("forbid_tools"):
        outcomes["no_forbidden_tools"] = not (set(case["forbid_tools"]) & set(names))
    if case.get("expect_state"):
        state = session.describe()
        outcomes["state_present"] = all(k in state for k in case["expect_state"])
    for check in case.get("checks", []):
        if check not in CHECKS:
            raise KeyError(f"Unknown check '{check}' in case {case['id']}")
        try:
            outcomes[check] = bool(CHECKS[check](session, result))
        except Exception:
            outcomes[check] = False
    if case.get("answer_regex"):
        outcomes["answer_regex"] = re.search(case["answer_regex"], result.text, re.I) is not None
    if case.get("max_steps"):
        outcomes["within_step_budget"] = result.steps <= case["max_steps"]
    outcomes["no_provider_error"] = result.stop_reason != "error"
    return outcomes


def run_case(case: dict[str, Any], provider: Provider, workspace: str | Path) -> dict[str, Any]:
    session = Session(workspace=Path(workspace) / "evals")
    agent = Agent(provider, session, max_steps=max(case.get("max_steps", 12), 1) + 2)
    t0 = time.perf_counter()
    result = agent.run(case["prompt"])
    seconds = round(time.perf_counter() - t0, 2)
    outcomes = grade(case, session, result)
    return {
        "id": case["id"],
        "passed": all(outcomes.values()),
        "outcomes": outcomes,
        "tools": result.tool_names,
        "steps": result.steps,
        "seconds": seconds,
        "usage": result.usage,
        "answer": result.text[:800],
        "errors": [e.result.get("error") for e in result.trace if e.is_error],
    }


def run_evals(
    provider: Provider,
    cases: str | Path | None = None,
    workspace: str | Path = "eegagent_workspace",
    verbose: bool = False,
) -> dict[str, Any]:
    all_cases = load_cases(cases)
    rows = []
    for case in all_cases:
        row = run_case(case, provider, workspace)
        rows.append(row)
        if verbose:
            print(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['id']}: {row['tools']} ({row['seconds']}s)")
    n_pass = sum(r["passed"] for r in rows)
    summary = {
        "provider": provider.name,
        "model": provider.model,
        "n_cases": len(rows),
        "n_passed": n_pass,
        "pass_rate": round(n_pass / max(len(rows), 1), 3),
        "total_usage": _sum_usage(rows),
        "cases": rows,
    }
    summary["report"] = format_report(summary)
    out_dir = Path(workspace) / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"results_{provider.name}_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    summary["results_path"] = str(out)
    return summary


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    tot: dict[str, int] = {}
    for r in rows:
        for k, v in r["usage"].items():
            tot[k] = tot.get(k, 0) + int(v)
    return tot


def format_report(summary: dict[str, Any]) -> str:
    lines = [f"eegagent eval: provider={summary['provider']} model={summary['model']}", ""]
    lines.append(f"{'case':<28} {'result':<6} {'steps':>5} {'time':>7}  failed checks")
    for r in summary["cases"]:
        failed = [k for k, v in r["outcomes"].items() if not v]
        lines.append(
            f"{r['id']:<28} {'PASS' if r['passed'] else 'FAIL':<6} {r['steps']:>5} {r['seconds']:>6}s  {', '.join(failed)}"
        )
    lines.append("")
    lines.append(
        f"passed {summary['n_passed']}/{summary['n_cases']} ({summary['pass_rate']:.0%}); tokens: {summary['total_usage']}"
    )
    return "\n".join(lines)

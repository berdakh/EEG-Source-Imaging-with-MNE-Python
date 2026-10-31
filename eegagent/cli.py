"""Command line: ``eegagent run|chat|eval|serve|ui``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import Agent
from .providers import PROVIDERS, get_provider
from .session import Session


def _printer(verbose: bool):
    def on_event(kind, payload):
        if kind == "tool_call":
            print(f"  -> {payload['name']}({json.dumps(payload['args'])})", file=sys.stderr)
        elif kind == "tool_result" and verbose:
            tag = "ERR" if payload.is_error else "ok"
            print(
                f"  <- {tag} {json.dumps(payload.result, default=str)[:400]} ({payload.seconds}s)",
                file=sys.stderr,
            )
        elif kind == "tool_result" and payload.is_error:
            print(f"  <- ERR {payload.result.get('error')}", file=sys.stderr)

    return on_event


def _agent(args) -> Agent:
    provider = get_provider(args.provider, model=args.model)
    session = Session(workspace=args.workspace)
    print(
        f"[eegagent] provider={provider.name} model={provider.model} session={session.session_id}",
        file=sys.stderr,
    )
    return Agent(provider, session, max_steps=args.max_steps, on_event=_printer(args.verbose))


def cmd_run(args) -> int:
    agent = _agent(args)
    res = agent.run(args.prompt)
    print(res.text)
    if args.json:
        print(
            json.dumps(
                {"tools": res.tool_names, "steps": res.steps, "usage": res.usage, "stop": res.stop_reason}
            )
        )
    return 0 if res.stop_reason != "error" else 1


def cmd_chat(args) -> int:
    agent = _agent(args)
    print("Type your request (empty line or Ctrl-D to quit).")
    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            break
        if not line:
            break
        res = agent.run(line)
        print(f"\nagent> {res.text}")
    return 0


def cmd_eval(args) -> int:
    from .evals.runner import run_evals

    provider = get_provider(args.provider, model=args.model)
    summary = run_evals(provider, cases=args.cases, workspace=args.workspace, verbose=args.verbose)
    print(json.dumps(summary, indent=2) if args.json else summary["report"])
    return 0 if summary["pass_rate"] >= args.min_pass_rate else 1


def cmd_serve(args) -> int:
    import uvicorn

    os.environ.setdefault("EEGAGENT_PROVIDER", args.provider or "anthropic")
    if args.model:
        os.environ["EEGAGENT_MODEL"] = args.model
    uvicorn.run("eegagent.server:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_ui(args) -> int:
    import subprocess

    from . import ui

    env = dict(os.environ, EEGAGENT_API=args.api)
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", ui.__file__, "--server.port", str(args.port)], env=env
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eegagent", description="LLM agent for EEG source imaging with MNE-Python."
    )
    p.add_argument(
        "--provider", choices=PROVIDERS, default=None, help="default: $EEGAGENT_PROVIDER or anthropic"
    )
    p.add_argument("--model", default=None, help="model id (default: provider default or $EEGAGENT_MODEL)")
    p.add_argument("--workspace", default="eegagent_workspace")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="one request, print the answer")
    s.add_argument("prompt")
    s.add_argument("--json", action="store_true", help="also print the tool trace summary as JSON")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("chat", help="interactive multi-turn session")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("eval", help="run the eval suite against a provider")
    s.add_argument("--cases", default=None, help="YAML file (default: built-in cases)")
    s.add_argument("--min-pass-rate", type=float, default=0.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("serve", help="start the HTTP API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("ui", help="start the Streamlit front end (expects the API to be running)")
    s.add_argument("--api", default="http://localhost:8000")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(fn=cmd_ui)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

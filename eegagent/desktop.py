"""Desktop launcher: start the API and the UI together and open the app in a window.

    eegagent desktop            # native window (pywebview) when available, else the browser
    eegagent-desktop            # same, as a GUI entry point for shortcuts and installers

Both services run as child processes of this launcher and are shut down when the window
closes. The launcher works unchanged when frozen with PyInstaller: children are spawned as
the frozen executable itself with a ``--serve-*`` flag.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import config_dir, default_workspace, setting

APP_TITLE = "EEG-Agent"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for(url: str, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def _child_command(role: str, port: int) -> list[str]:
    """argv for a child process, valid both from source and from a PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        return [sys.executable, f"--serve-{role}", str(port)]
    return [sys.executable, "-m", "eegagent.desktop", f"--serve-{role}", str(port)]


def _spawn(role: str, port: int, env: dict[str, str], log_dir: Path) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{role}.log", "ab")  # noqa: SIM115 - kept open for the child's lifetime
    kwargs: dict = {"env": env, "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":  # no console windows for children
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(_child_command(role, port), **kwargs)


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + 10
    for p in procs:
        while p.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if p.poll() is None:
            p.kill()


# ---------------------------------------------------------------------------- child roles


def serve_api(port: int) -> None:
    import uvicorn

    uvicorn.run("eegagent.server:app", host="127.0.0.1", port=port, log_level="info")


def serve_ui(port: int) -> None:
    from streamlit.web import bootstrap

    ui_path = str(Path(__file__).with_name("ui.py"))
    flags = {
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }
    bootstrap.load_config_options(flag_options=flags)  # what `streamlit run` does before run()
    bootstrap.run(ui_path, False, [], flags)


# ---------------------------------------------------------------------------- launcher


def launch(
    provider: str | None = None,
    model: str | None = None,
    no_window: bool = False,
    smoke: bool = False,
    api_port: int | None = None,
    ui_port: int | None = None,
) -> int:
    api_port = api_port or free_port()
    ui_port = ui_port or free_port()
    api_url = f"http://127.0.0.1:{api_port}"
    ui_url = f"http://127.0.0.1:{ui_port}"

    env = dict(os.environ)
    env["EEGAGENT_API"] = api_url
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("EEGAGENT_WORKSPACE", str(setting("workspace", None, str(default_workspace()))))
    if provider:
        env["EEGAGENT_PROVIDER"] = provider
    if model:
        env["EEGAGENT_MODEL"] = model
    Path(env["EEGAGENT_WORKSPACE"]).mkdir(parents=True, exist_ok=True)
    log_dir = config_dir() / "logs"

    procs: list[subprocess.Popen] = []
    try:
        procs.append(_spawn("api", api_port, env, log_dir))
        procs.append(_spawn("ui", ui_port, env, log_dir))
        if not wait_for(f"{api_url}/health"):
            print(f"API did not start; see {log_dir / 'api.log'}", file=sys.stderr)
            return 1
        if not wait_for(f"{ui_url}/_stcore/health"):
            print(f"UI did not start; see {log_dir / 'ui.log'}", file=sys.stderr)
            return 1
        print(f"{APP_TITLE}: UI at {ui_url}, API at {api_url} (logs in {log_dir})", file=sys.stderr)
        if smoke:
            return 0
        if not no_window and _open_native_window(ui_url):
            return 0
        import webbrowser

        webbrowser.open(ui_url)
        print("Running in the browser. Press Ctrl-C to quit.", file=sys.stderr)
        _wait_for_interrupt(procs)
        return 0
    finally:
        _stop(procs)


def _open_native_window(url: str) -> bool:
    """Blocks until the window is closed. Returns False if pywebview is unavailable."""
    try:
        import webview
    except Exception:
        return False
    try:
        webview.create_window(APP_TITLE, url, width=1440, height=960, min_size=(900, 600))
        webview.start()
        return True
    except Exception as e:  # missing GTK/WebView2 etc.: fall back to the browser
        print(f"Native window unavailable ({e}); opening the browser instead.", file=sys.stderr)
        return False


def _wait_for_interrupt(procs: list[subprocess.Popen]) -> None:
    stop = False

    def handler(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    while not stop and all(p.poll() is None for p in procs):
        time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eegagent-desktop", description=f"{APP_TITLE} desktop launcher")
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--smoke", action="store_true", help="start both services, check them, exit")
    p.add_argument("--serve-api", type=int, metavar="PORT", help=argparse.SUPPRESS)
    p.add_argument("--serve-ui", type=int, metavar="PORT", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.serve_api:
        serve_api(args.serve_api)
        return 0
    if args.serve_ui:
        serve_ui(args.serve_ui)
        return 0
    return launch(provider=args.provider, model=args.model, no_window=args.no_window, smoke=args.smoke)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

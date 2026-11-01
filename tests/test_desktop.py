import subprocess
import sys

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("uvicorn")

from eegagent import desktop  # noqa: E402


def test_child_command_from_source():
    cmd = desktop._child_command("api", 1234)
    assert cmd[:3] == [sys.executable, "-m", "eegagent.desktop"] and cmd[-2:] == ["--serve-api", "1234"]


def test_free_port_is_free():
    import socket

    port = desktop.free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_smoke_launch_starts_both_services(tmp_path, monkeypatch):
    monkeypatch.setenv("EEGAGENT_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("EEGAGENT_PROVIDER", "rule")
    rc = desktop.launch(no_window=True, smoke=True)
    assert rc == 0
    assert (tmp_path / "logs" / "api.log").exists() and (tmp_path / "logs" / "ui.log").exists()
    assert (tmp_path / "workspace").is_dir()


def test_cli_entry_point_smoke(tmp_path):
    env = {
        "EEGAGENT_CONFIG": str(tmp_path / "config.json"),
        "EEGAGENT_PROVIDER": "rule",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    out = subprocess.run(
        [sys.executable, "-m", "eegagent.cli", "desktop", "--smoke", "--no-window"],
        env={**env, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "UI at http://127.0.0.1:" in out.stderr

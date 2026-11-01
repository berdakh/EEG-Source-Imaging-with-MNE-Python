"""User configuration for the desktop app and CLI.

Precedence for every setting: explicit argument > environment variable > config file > default.
The config file lives at ``~/.eegagent/config.json`` (override with ``EEGAGENT_CONFIG``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV_FOR_KEY = {
    "provider": "EEGAGENT_PROVIDER",
    "model": "EEGAGENT_MODEL",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "ollama_host": "OLLAMA_HOST",
    "workspace": "EEGAGENT_WORKSPACE",
}

DEFAULTS: dict[str, Any] = {
    "provider": "anthropic",
    "model": "",
    "anthropic_api_key": "",
    "ollama_host": "http://localhost:11434",
    "workspace": "",
}

SECRET_KEYS = {"anthropic_api_key"}


def config_dir() -> Path:
    return config_path().parent


def config_path() -> Path:
    return Path(os.environ.get("EEGAGENT_CONFIG") or Path.home() / ".eegagent" / "config.json").expanduser()


def default_workspace() -> Path:
    return config_dir() / "workspace"


def load_config() -> dict[str, Any]:
    """Config file contents merged over the defaults. Missing or broken file gives the defaults."""
    data = dict(DEFAULTS)
    path = config_path()
    try:
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update({k: v for k, v in loaded.items() if k in DEFAULTS})
    except (OSError, json.JSONDecodeError):
        pass
    return data


def save_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the config file (created with owner-only permissions) and return it."""
    data = load_config()
    data.update({k: v for k, v in updates.items() if k in DEFAULTS and v is not None})
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - Windows
        pass
    return data


def setting(key: str, explicit: Any = None, default: Any = None) -> Any:
    """Resolve one setting with the documented precedence. Empty strings count as unset."""
    if explicit not in (None, ""):
        return explicit
    env = os.environ.get(ENV_FOR_KEY[key])
    if env:
        return env
    value = load_config().get(key)
    if value not in (None, ""):
        return value
    return default if default is not None else DEFAULTS.get(key)


def public_settings() -> dict[str, Any]:
    """Settings safe to show in a UI: secrets are reported as present/absent only."""
    out: dict[str, Any] = {}
    for key in DEFAULTS:
        value = setting(key)
        if key in SECRET_KEYS:
            out[f"has_{key}"] = bool(value)
        else:
            out[key] = value or ""
    out["config_path"] = str(config_path())
    return out

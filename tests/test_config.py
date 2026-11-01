import json

import pytest

from eegagent import config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("EEGAGENT_CONFIG", str(tmp_path / "cfg" / "config.json"))
    for var in (
        "EEGAGENT_PROVIDER",
        "EEGAGENT_MODEL",
        "ANTHROPIC_API_KEY",
        "OLLAMA_HOST",
        "EEGAGENT_WORKSPACE",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_defaults_without_file():
    assert config.load_config() == config.DEFAULTS
    assert config.setting("provider") == "anthropic"
    assert config.setting("ollama_host") == "http://localhost:11434"


def test_save_and_precedence(monkeypatch):
    saved = config.save_config(
        {"provider": "ollama", "model": "qwen3:8b", "anthropic_api_key": "sk-test", "bogus": 1}
    )
    assert saved["provider"] == "ollama" and "bogus" not in saved
    assert json.loads(config.config_path().read_text())["model"] == "qwen3:8b"
    assert config.setting("model") == "qwen3:8b"
    monkeypatch.setenv("EEGAGENT_MODEL", "from-env")
    assert config.setting("model") == "from-env", "env beats file"
    assert config.setting("model", "explicit") == "explicit", "explicit beats env"
    assert config.setting("model", "") == "from-env", "empty explicit counts as unset"


def test_public_settings_masks_secret():
    config.save_config({"anthropic_api_key": "sk-secret"})
    pub = config.public_settings()
    assert pub["has_anthropic_api_key"] is True
    assert "sk-secret" not in json.dumps(pub)
    assert pub["config_path"].endswith("config.json")


def test_broken_file_falls_back_to_defaults():
    path = config.config_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert config.load_config()["provider"] == "anthropic"


def test_providers_read_config():
    from eegagent.providers import get_provider
    from eegagent.providers.ollama_provider import OllamaProvider

    config.save_config({"provider": "ollama", "model": "llama3.1", "ollama_host": "http://box:11434/"})
    p = get_provider()
    assert isinstance(p, OllamaProvider) and p.model == "llama3.1" and p.host == "http://box:11434"
    assert get_provider("rule").name == "rule", "explicit name wins over the file"

import pytest
import yaml

from eegmodels.registry import RegistryError, get_spec, list_models, load_registry, parse_spec


def minimal(**over):
    d = {
        "name": "toy",
        "status": "available",
        "source": {"type": "hf", "repo_id": "org/toy"},
        "input": {"sfreq": 100},
        "adapter": {"class": "planned"},
    }
    d.update(over)
    return d


def test_parse_minimal_entry():
    s = parse_spec(minimal())
    assert s.name == "toy" and s.family == "toy"
    assert s.source.revision == "main"
    assert s.input.units == "uV"
    assert s.license.spdx == "unknown"
    assert s.available


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "nope"},
        {"source": {"type": "ftp"}},
        {"source": {"type": "hf"}},
        {"source": {"type": "url", "files": []}},
        {"source": {"type": "url", "files": [{"url": "x"}]}},
        {"input": {"sfreq": 100, "units": "mV"}},
        {"adapter": {}},
    ],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(RegistryError):
        parse_spec(minimal(**bad))


def test_builtin_registry_is_valid():
    specs = load_registry()
    assert len(specs) >= 5
    for s in specs.values():
        assert s.path is not None and s.path.stem == s.name, "file name should match entry name"
        from eegmodels.adapters import get_adapter_class

        get_adapter_class(s.adapter.cls)


def test_unknown_name_gives_hint():
    with pytest.raises(RegistryError, match="Did you mean"):
        get_spec("steegformer")


def test_extra_registry_dir_overrides(tmp_path, monkeypatch):
    (tmp_path / "toy.yaml").write_text(yaml.safe_dump(minimal()))
    monkeypatch.setenv("EEGMODELS_REGISTRY", str(tmp_path))
    assert "toy" in list_models()
    assert get_spec("toy").path == tmp_path / "toy.yaml"

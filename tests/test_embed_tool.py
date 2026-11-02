"""The eegmodels bridge, exercised with an echo adapter and a fake registry entry (no weights, no network)."""

import numpy as np
import pytest
import yaml

from eegagent.session import Session
from eegagent.tools import TOOLS, run_tool
from eegmodels.adapters import ADAPTERS
from eegmodels.adapters.base import BaseAdapter


class EchoAdapter(BaseAdapter):
    def load_weights(self):
        self.model = "echo"

    def forward(self, batch):
        return batch.mean(axis=-1)  # (n_windows, n_channels) "embedding"


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "echo-test.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "echo-test",
                "status": "available",
                "source": {"type": "hf", "repo_id": "test/echo"},
                "license": {"spdx": "MIT"},
                "input": {
                    "sfreq": 100,
                    "units": "uV",
                    "window_seconds": 0.7,
                    "channels": ["Cz", "C3", "Fp1"],
                },
                "adapter": {"class": "echo"},
            }
        )
    )
    cache = tmp_path / "cache" / "echo-test"
    cache.mkdir(parents=True)
    (cache / "weights.bin").write_bytes(b"x")  # marks the model as already downloaded
    monkeypatch.setenv("EEGMODELS_REGISTRY", str(registry))
    monkeypatch.setenv("EEGMODELS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setitem(ADAPTERS, "echo", EchoAdapter)
    return "echo-test"


def test_tool_is_registered_with_model_names():
    assert "embed_with_foundation_model" in TOOLS
    assert "steegformer-base" in TOOLS["embed_with_foundation_model"].description


def test_embed_epochs(tmp_path, fake_model):
    session = Session(workspace=tmp_path / "ws")
    run_tool(session, "load_recording", {"source": "simulated"})
    run_tool(session, "make_epochs", {"tmin": -0.2, "tmax": 0.5, "reject_uv": None})
    r, err = run_tool(session, "embed_with_foundation_model", {"model": fake_model})
    assert not err, r
    assert r["shape"] == [30, 3] and r["input"] == "epochs"
    z = np.load(r["path"])
    assert z.shape == (30, 3)
    assert session.describe()["embeddings"]["model"] == fake_model


def test_embed_raw_windows(tmp_path, fake_model):
    session = Session(workspace=tmp_path / "ws")
    run_tool(session, "load_recording", {"source": "simulated"})
    r, err = run_tool(session, "embed_with_foundation_model", {"model": fake_model})
    assert not err, r
    assert r["input"] == "raw" and r["shape"][0] == int(60 // 0.7) and r["shape"][1] == 3


def test_embed_errors_are_tool_errors(tmp_path, fake_model):
    session = Session(workspace=tmp_path / "ws")
    r, err = run_tool(session, "embed_with_foundation_model", {"model": fake_model})
    assert err and "load_recording" in r["error"]
    run_tool(session, "load_recording", {"source": "simulated"})
    r, err = run_tool(session, "embed_with_foundation_model", {"model": "no-such-model"})
    assert err and "Unknown model" in r["error"]

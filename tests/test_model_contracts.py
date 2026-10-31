"""Offline contract/feature-routing tests. No weights or Braindecode runtime are loaded."""

from contextlib import nullcontext

import numpy as np
import pytest
from test_adapters import EchoAdapter, spec

from eegmodels.adapters.base import AdapterError
from eegmodels.adapters.braindecode import BraindecodeAdapter
from eegmodels.api import load
from eegmodels.registry import RegistryError, load_registry


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch:
    float32 = np.float32

    @staticmethod
    def as_tensor(batch, **kwargs):
        return batch

    @staticmethod
    def no_grad():
        return nullcontext()


class FeatureModel:
    n_chans = 2

    def forward(self, x, return_features=False):
        assert return_features is True, "Never silently request logits"
        return {"features": FakeTensor(np.ones((len(x), 16))), "cls_token": None}

    __call__ = forward


def feature_adapter(model=None):
    adapter = BraindecodeAdapter(spec(window_seconds=None), "/unused")
    adapter.model = FeatureModel() if model is None else model
    adapter._torch = FakeTorch()
    return adapter


def test_embed_requests_encoder_features_not_logits():
    features = feature_adapter().embed(np.zeros((2, 100)), sfreq=100)
    assert features.shape == (1, 16)
    assert np.all(features == 1)


def test_predict_refuses_unvalidated_task_head():
    with pytest.raises(AdapterError, match="trained head"):
        feature_adapter().predict(np.zeros((2, 100)), sfreq=100)


def test_old_runtime_does_not_fall_back_to_logits():
    class OldModel:
        def forward(self, x):
            raise AssertionError("Must be rejected before inference")

    with pytest.raises(AdapterError, match="lacks return_features"):
        feature_adapter(OldModel()).embed(np.zeros((2, 100)), sfreq=100)


def test_wrong_montage_count_is_rejected():
    with pytest.raises(AdapterError, match="constructed for 2 channels"):
        feature_adapter().embed(np.zeros((3, 100)), sfreq=100)


@pytest.mark.parametrize(
    "output",
    [
        FakeTensor([[1, 2]]),
        {},
        {"features": None},
        {"features": FakeTensor([[np.nan]])},
        {"features": FakeTensor(np.ones((2, 3)))},
    ],
)
def test_invalid_feature_contract_is_rejected(output):
    class BrokenModel:
        def forward(self, x, return_features=False):
            return output

        __call__ = forward

    with pytest.raises(AdapterError):
        feature_adapter(BrokenModel()).embed(np.zeros((2, 100)), sfreq=100)


def test_unverified_contract_fails_before_weights_or_data_transform():
    adapter = EchoAdapter(spec(contract_verified=False, validation_notes="montage unresolved"), "/unused")
    with pytest.raises(AdapterError, match="montage unresolved"):
        adapter.embed(np.zeros((2, 100)), sfreq=100)
    assert adapter.model is None


def test_load_unverified_contract_does_not_download(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unverified inference should fail before download")

    monkeypatch.setattr("eegmodels.api.pull", forbidden)
    with pytest.raises(AdapterError, match="not validated"):
        load("steegformer-base")


def test_fixed_checkpoint_channels_cannot_be_loose():
    adapter = BraindecodeAdapter(spec(channels=["Cz", "Fp1"]), "/unused")
    with pytest.raises(AdapterError, match="strict_channels"):
        adapter.prepare(np.zeros((1, 100)), sfreq=100, ch_names=["Cz"], strict_channels=False)


def test_duplicate_channel_names_are_not_silently_selected():
    adapter = EchoAdapter(spec(channels=["Cz"]), "/unused")
    with pytest.raises(AdapterError, match="duplicate"):
        adapter.prepare(np.zeros((2, 100)), sfreq=100, ch_names=["Cz", "EEG CZ-REF"])


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_invalid_input_rate_rejected(rate):
    with pytest.raises(AdapterError, match="sfreq"):
        EchoAdapter(spec(), "/unused").prepare(np.zeros((2, 100)), sfreq=rate)


def test_nonfinite_input_rejected():
    with pytest.raises(AdapterError, match="non-finite"):
        EchoAdapter(spec(), "/unused").prepare(np.full((2, 100), np.nan), sfreq=100)


def test_registry_contract_corrections():
    specs = load_registry()
    assert all(not entry.input.contract_verified for entry in specs.values())
    for name in ("steegformer-base", "steegformer-large"):
        assert specs[name].input.channel_vocab_size == 145
        assert specs[name].input.sfreq is None
        assert specs[name].input.n_channels_max is None
    assert specs["steegformer-largev2"].input.channel_vocab_size == 256
    mi = specs["mirepnet"].input
    assert len(mi.channels) == len(set(mi.channels)) == 45
    assert mi.sfreq == 250 and mi.window_seconds == 4
    assert specs["brainbert"].input.n_channels_max is None
    assert specs["mvpformer-s"].input.segment_seconds == 5
    assert specs["mvpformer-s"].input.window_seconds is None
    assert specs["labram-base"].input.amplitude_clip_uv is None
    assert specs["reve-base"].status == "planned"


@pytest.mark.parametrize(
    "change",
    [
        {"sfreq": None},
        {"sfreq": float("nan")},
        {"sfreq": 0},
        {"window_seconds": -1},
        {"channel_vocab_size": 1.5},
        {"contract_verified": "false"},
        {"preprocessing": "not-a-list"},
        {"channels": ["Cz", "cz"]},
    ],
)
def test_registry_rejects_invalid_contract(change):
    with pytest.raises(RegistryError):
        spec(**change)


def test_unknown_rate_is_explicit_and_blocks_inference():
    entry = spec(sfreq=None, contract_verified=False)
    assert entry.input.sfreq is None
    with pytest.raises(AdapterError, match="not validated"):
        EchoAdapter(entry, "/unused").prepare(np.zeros((2, 100)), sfreq=100)


def test_legacy_registry_without_contract_flag_is_not_trusted():
    from eegmodels.registry import parse_spec

    entry = parse_spec(
        {
            "name": "legacy",
            "status": "available",
            "source": {"type": "hf", "repo_id": "org/model"},
            "input": {"sfreq": 100},
            "adapter": {"class": "planned"},
        }
    )
    assert entry.input.contract_verified is False
    with pytest.raises(AdapterError, match="not validated"):
        EchoAdapter(entry, "/unused").prepare(np.zeros((2, 100)), sfreq=100)


def test_duplicate_required_aliases_are_not_duplicated():
    adapter = EchoAdapter(spec(channels=["Cz", "EEG CZ-REF"]), "/unused")
    with pytest.raises(AdapterError, match="duplicate required"):
        adapter.prepare(np.zeros((1, 100)), sfreq=100, ch_names=["Cz"])

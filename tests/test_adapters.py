import numpy as np
import pytest

from eegmodels.adapters.base import AdapterError, BaseAdapter, resample, window
from eegmodels.registry import parse_spec


class EchoAdapter(BaseAdapter):
    def load_weights(self):
        self.model = "echo"

    def forward(self, batch):
        return batch


def spec(**inp):
    base = {"sfreq": 100, "units": "uV", "window_seconds": 1.0}
    base.update(inp)
    return parse_spec(
        {
            "name": "echo",
            "status": "available",
            "source": {"type": "hf", "repo_id": "x/y"},
            "input": base,
            "adapter": {"class": "planned"},
        }
    )


def test_windowing_and_resampling():
    ad = EchoAdapter(spec(), weights_dir="/nonexistent")
    x = np.random.randn(4, 1000).astype(np.float32)  # 4 ch, 5 s at 200 Hz
    out = ad.embed(x, sfreq=200)
    assert out.shape == (5, 4, 100)


def test_channel_selection_case_insensitive():
    ad = EchoAdapter(spec(channels=["Fp1", "Cz"]), weights_dir="/x")
    x = np.arange(3 * 100, dtype=np.float32).reshape(3, 100)
    out = ad.prepare(x, sfreq=100, ch_names=["EEG CZ-REF", "O1", "fp1"])
    assert out.shape == (1, 2, 100)
    np.testing.assert_array_equal(out[0, 0], x[2])  # Fp1
    np.testing.assert_array_equal(out[0, 1], x[0])  # Cz


def test_missing_channel_strict_vs_loose():
    ad = EchoAdapter(spec(channels=["Fp1", "Cz"]), weights_dir="/x")
    x = np.zeros((1, 100), dtype=np.float32)
    with pytest.raises(AdapterError, match="missing channels"):
        ad.prepare(x, sfreq=100, ch_names=["Cz"])
    out = ad.prepare(x, sfreq=100, ch_names=["Cz"], strict_channels=False)
    assert out.shape == (1, 1, 100)


def test_units_conversion_and_clipping():
    ad = EchoAdapter(spec(units="V", amplitude_clip_uv=50), weights_dir="/x")
    x = np.full((1, 100), 200.0, dtype=np.float32)  # 200 uV
    out = ad.prepare(x, sfreq=100)
    assert np.allclose(out, 50e-6)  # clipped to 50 uV then expressed in volts


def test_user_array_in_volts():
    ad = EchoAdapter(spec(), weights_dir="/x")
    x = np.full((1, 100), 1e-6, dtype=np.float32)
    out = ad.prepare(x, sfreq=100, units="V")
    assert np.allclose(out, 1.0)  # 1 uV in model units (uV)


def test_epochs_wrong_length_rejected():
    ad = EchoAdapter(spec(), weights_dir="/x")
    with pytest.raises(AdapterError, match="window"):
        ad.prepare(np.zeros((2, 3, 50), dtype=np.float32), sfreq=100)


def test_array_needs_sfreq():
    ad = EchoAdapter(spec(), weights_dir="/x")
    with pytest.raises(AdapterError, match="sfreq"):
        ad.prepare(np.zeros((1, 100)))


def test_window_too_short():
    with pytest.raises(AdapterError, match="shorter"):
        window(np.zeros((1, 10)), 100, 1.0)


def test_resample_numpy_fallback_length():
    out = resample(np.random.randn(2, 400).astype(np.float32), 200, 100)
    assert out.shape == (2, 200)


def test_mne_raw_roundtrip():
    mne = pytest.importorskip("mne")
    info = mne.create_info(["Fp1", "Cz", "EOG1"], 100, ["eeg", "eeg", "eog"])
    data = np.full((3, 300), 5e-6)
    raw = mne.io.RawArray(data, info, verbose="error")
    ad = EchoAdapter(spec(channels=["Cz", "Fp1"]), weights_dir="/x")
    out = ad.prepare(raw)
    assert out.shape == (3, 2, 100)
    assert np.allclose(out, 5.0)  # volts -> microvolts

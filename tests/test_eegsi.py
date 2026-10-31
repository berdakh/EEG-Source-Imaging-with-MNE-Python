"""End-to-end tests of the eegsi library on synthetic data (no downloads)."""

import numpy as np
import pytest

from eegsi import data, forward, inverse, preprocessing, report, stats


@pytest.fixture(scope="module")
def raw():
    return data.simulate_raw(duration=40.0, n_events=20, seed=1)


@pytest.fixture(scope="module")
def epochs(raw):
    r = preprocessing.apply_filters(raw, 1.0, 30.0, notch=None)
    return preprocessing.make_epochs(r, tmin=-0.2, tmax=0.4, reject_uv=None)


def test_simulated_raw_has_montage_and_events(raw):
    assert raw.get_montage() is not None
    ev = data.find_events(raw)
    assert len(ev) == 20 and set(ev[:, 2]) == {1, 2}


def test_filters_and_notch(raw):
    out = preprocessing.apply_filters(raw, 1.0, 30.0, notch=50.0)
    assert out.info["highpass"] >= 1.0 and out.info["lowpass"] <= 30.0
    assert out is not raw


def test_id_outliers_flags_high_variance_row():
    X = np.random.default_rng(0).standard_normal((10, 500))
    X[3] *= 20
    good, bad = preprocessing.id_outliers(X, threshold=(None, 2.0))
    assert bad == [3] and len(good) == 9


def test_find_bad_channels_and_interpolate(raw):
    noisy = raw.copy()
    noisy._data[noisy.ch_names.index("T7")] *= 50
    bads = preprocessing.find_bad_channels(noisy)
    assert "T7" in bads
    fixed = preprocessing.mark_and_interpolate(noisy, bads)
    assert fixed.info["bads"] == []
    assert np.std(fixed.get_data(picks="T7")) < np.std(noisy.get_data(picks="T7")) / 10


def test_epochs_and_evoked(epochs):
    assert set(epochs.event_id) == {"event_1", "event_2"}
    ev = epochs["event_1"].average()
    peak_ch, peak_t = ev.get_peak(ch_type="eeg", tmin=0.05, tmax=0.15)
    assert peak_ch in {"C3", "Cz", "FC1", "CP1"}
    assert 0.05 <= peak_t <= 0.15


def test_average_reference(epochs):
    ref = preprocessing.set_average_reference(epochs)
    assert abs(ref.get_data(picks="eeg").mean(axis=1)).max() < 1e-9


def test_sphere_forward_inverse_recovers_left_central_source(epochs):
    fwd = forward.make_sphere_forward(epochs.info, pos=0.02)
    d = forward.describe_forward(fwd)
    assert d["source_space"] == "volume" and d["n_sensors"] == 32
    cov = inverse.compute_noise_covariance(epochs, tmax=0.0)
    inv = inverse.make_inverse(epochs.info, fwd, cov, loose=1.0, depth=None)
    stc = inverse.apply_inverse_evoked(epochs["event_1"].average(), inv, method="dSPM")
    peak = inverse.peak_source(stc, fwd, tmin=0.05, tmax=0.15)
    assert 0.05 <= peak["time_s"] <= 0.15
    x, y, z = peak["position_mm"]
    assert x < 0, "C3/FC1/CP1 bump should localise to the left hemisphere"
    assert z > 0, "and above the head centre"
    s = inverse.stc_summary(stc)
    assert s["n_sources"] == d["n_sources"]


def test_cluster_test_finds_condition_difference(epochs):
    res = stats.sensor_cluster_test(epochs, "event_1", "event_2", n_permutations=200)
    assert res["n_clusters"] >= 1
    assert res["n_significant"] >= 1
    top = res["significant_clusters"][0]
    assert top["tmin_s"] <= 0.1 <= top["tmax_s"] + 0.05
    with pytest.raises(ValueError, match="Unknown condition"):
        stats.sensor_cluster_test(epochs, "event_1", "nope")


def test_report_builds_html(tmp_path, epochs):
    ev = epochs["event_1"].average()
    out = report.build_report(
        tmp_path / "r.html",
        raw=None,
        epochs=epochs,
        evoked=ev,
        extras={"peak": {"t": 0.1}},
        log=["loaded", "filtered"],
    )
    text = out.read_text()
    assert "Evoked" in text and "Processing log" in text and "filtered" in text

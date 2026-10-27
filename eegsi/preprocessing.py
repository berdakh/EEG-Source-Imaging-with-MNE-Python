"""Preprocessing: filtering, bad-channel detection, referencing, epoching, ICA."""

from __future__ import annotations

import mne
import numpy as np

from .data import find_events


def apply_filters(
    raw: mne.io.BaseRaw,
    l_freq: float | None = 0.1,
    h_freq: float | None = 40.0,
    notch: float | None = None,
) -> mne.io.BaseRaw:
    """Zero-phase FIR band-pass and optional notch (with harmonics up to h_freq or Nyquist)."""
    raw = raw.copy().load_data()
    if notch:
        top = h_freq or raw.info["sfreq"] / 2
        freqs = np.arange(notch, min(top, raw.info["sfreq"] / 2), notch)
        if len(freqs):
            raw.notch_filter(freqs, verbose="error")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, phase="zero", verbose="error")
    return raw


def id_outliers(
    X: np.ndarray,
    threshold: tuple[float | None, float | None] = (None, 2.0),
    max_iter: int = 2,
    feat: str = "var",
) -> tuple[list[int], list[int]]:
    """Iterative z-score outlier detection on per-row features (port of the legacy idOutliers).

    Returns (good_rows, bad_rows). ``threshold`` is (lower, upper) in standard deviations;
    ``None`` disables that side. Rows flagged in one pass are excluded from the next.
    """
    if feat == "var":
        values = np.var(X, axis=1)
    elif feat == "mean":
        values = np.mean(np.abs(X), axis=1)
    else:
        raise ValueError("feat must be 'var' or 'mean'")
    good = np.ones(len(values), dtype=bool)
    lo, hi = threshold
    for _ in range(max_iter):
        mu, sd = values[good].mean(), values[good].std()
        if sd == 0:
            break
        z = (values - mu) / sd
        new_bad = np.zeros_like(good)
        if hi is not None:
            new_bad |= z > hi
        if lo is not None:
            new_bad |= z < -lo
        new_bad &= good
        if not new_bad.any():
            break
        good &= ~new_bad
    return list(np.flatnonzero(good)), list(np.flatnonzero(~good))


def find_bad_channels(
    inst: mne.io.BaseRaw | mne.Epochs,
    threshold: float = 3.0,
    max_iter: int = 2,
    feat: str = "var",
) -> list[str]:
    """Channels whose variance is an outlier across EEG channels."""
    picks = mne.pick_types(inst.info, eeg=True, exclude=[])
    data = inst.get_data(picks=picks)
    if data.ndim == 3:  # epochs: (n_epochs, n_ch, n_times) -> (n_ch, n_epochs * n_times)
        data = np.transpose(data, (1, 0, 2)).reshape(len(picks), -1)
    _, bad_idx = id_outliers(data, threshold=(None, threshold), max_iter=max_iter, feat=feat)
    return [inst.ch_names[picks[i]] for i in bad_idx]


def mark_and_interpolate(inst, bads: list[str]):
    """Set ``info['bads']`` and interpolate them with spherical splines (needs a montage)."""
    inst = inst.copy()
    inst.info["bads"] = list(dict.fromkeys(list(inst.info["bads"]) + list(bads)))
    if inst.info["bads"]:
        inst.interpolate_bads(reset_bads=True, verbose="error")
    return inst


def set_average_reference(inst, projection: bool = True):
    inst = inst.copy()
    inst.set_eeg_reference("average", projection=projection, verbose="error")
    if projection:
        inst.apply_proj(verbose="error")
    return inst


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = -0.2,
    tmax: float = 0.5,
    event_id: dict[str, int] | int | None = None,
    baseline: tuple[float | None, float | None] | None = (None, 0),
    reject_uv: float | None = 150.0,
) -> mne.Epochs:
    """Segment around events found in the stim channel or annotations."""
    events = find_events(raw)
    if len(events) == 0:
        raise ValueError("No events found in the recording (no stim channel values, no annotations).")
    if event_id is None:
        ids = sorted(set(events[:, 2].tolist()))
        event_id = {f"event_{i}": i for i in ids}
    reject = {"eeg": reject_uv * 1e-6} if reject_uv else None
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        picks="eeg",
        verbose="error",
    )
    if len(epochs) == 0:
        raise ValueError("All epochs were rejected; raise reject_uv or check the data.")
    return epochs


def run_ica(
    epochs: mne.Epochs,
    n_components: int = 15,
    eog_channel: str | None = "Fp1",
    random_state: int = 23,
) -> tuple[mne.Epochs, list[int]]:
    """Fit FastICA and remove components correlated with a frontal channel used as EOG proxy."""
    from mne.preprocessing import ICA

    n_components = min(n_components, len(mne.pick_types(epochs.info, eeg=True)) - 1)
    ica = ICA(n_components=n_components, method="fastica", random_state=random_state, max_iter="auto")
    ica.fit(epochs, decim=3, verbose="error")
    excluded: list[int] = []
    if eog_channel and eog_channel in epochs.ch_names:
        inds, _ = ica.find_bads_eog(epochs, ch_name=eog_channel, threshold=3.0, verbose="error")
        excluded = list(inds)
    ica.exclude = excluded
    cleaned = ica.apply(epochs.copy(), verbose="error")
    return cleaned, excluded

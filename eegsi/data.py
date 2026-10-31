"""Loading recordings and generating synthetic data for tests and demos."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

STANDARD_32 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
    "AF3",
    "AF4",
    "PO3",
    "PO4",
]


def load_recording(path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    """Read any raw file MNE understands (BrainVision .vhdr, .fif, .edf, .set, ...)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such recording: {path}")
    raw = mne.io.read_raw(path, preload=preload, verbose="error")
    return raw


def load_sample_raw(preload: bool = True) -> mne.io.BaseRaw:
    """The MNE sample dataset (downloads ~1.5 GB on first use), EEG channels only."""
    data_path = mne.datasets.sample.data_path(verbose="error")
    fname = Path(data_path) / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    raw = mne.io.read_raw_fif(fname, preload=preload, verbose="error")
    raw.pick(["eeg", "stim"])
    return raw


def simulate_raw(
    n_channels: int = 32,
    sfreq: float = 250.0,
    duration: float = 60.0,
    n_events: int = 30,
    event_ids: tuple[int, ...] = (1, 2),
    peak_channels: tuple[str, ...] = ("C3", "Cz", "FC1", "CP1"),
    peak_latency: float = 0.1,
    peak_amplitude_uv: float = 8.0,
    noise_uv: float = 3.0,
    seed: int = 0,
) -> mne.io.RawArray:
    """Synthetic EEG with a standard 10-20 montage, a stim channel and an evoked response.

    The response is a Gaussian bump of ``peak_amplitude_uv`` on ``peak_channels`` at
    ``peak_latency`` after each event, on top of pink-ish noise. Event id 2 gets half
    the amplitude so two conditions differ. Good enough to exercise every stage of
    the pipeline without downloading anything.
    """
    rng = np.random.default_rng(seed)
    ch_names = STANDARD_32[:n_channels]
    n_times = int(duration * sfreq)
    # 1/f (pink) noise shaped in the frequency domain, unit variance per channel
    white = rng.standard_normal((n_channels, n_times))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    spectrum[:, 1:] /= np.sqrt(freqs[1:])
    spectrum[:, 0] = 0
    pink = np.fft.irfft(spectrum, n=n_times, axis=1)
    pink /= np.std(pink, axis=1, keepdims=True)
    # per-channel noise gain so channel variances differ as in real recordings
    gains = rng.uniform(0.7, 1.3, size=(n_channels, 1))
    data = pink * gains * noise_uv * 1e-6

    stim = np.zeros((1, n_times))
    onsets = np.linspace(2 * sfreq, n_times - 2 * sfreq, n_events).astype(int)
    t = np.arange(n_times) / sfreq
    picks = [ch_names.index(c) for c in peak_channels if c in ch_names]
    for i, onset in enumerate(onsets):
        eid = event_ids[i % len(event_ids)]
        stim[0, onset] = eid
        amp = peak_amplitude_uv * 1e-6 * (1.0 if eid == event_ids[0] else 0.5)
        bump = amp * np.exp(-0.5 * ((t - (onset / sfreq + peak_latency)) / 0.03) ** 2)
        data[picks] += bump[None, :]

    info = mne.create_info(ch_names + ["STI 014"], sfreq, ["eeg"] * n_channels + ["stim"])
    raw = mne.io.RawArray(np.vstack([data, stim]), info, verbose="error")
    raw.set_montage("standard_1020", verbose="error")
    return raw


def find_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """Events from a stim channel if present, otherwise from annotations."""
    stim = mne.pick_types(raw.info, stim=True)
    if len(stim):
        return mne.find_events(raw, verbose="error")
    events, _ = mne.events_from_annotations(raw, verbose="error")
    return events

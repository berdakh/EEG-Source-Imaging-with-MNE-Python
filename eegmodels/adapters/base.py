"""Adapter base class and the input-contract enforcement shared by every model.

An adapter turns whatever the user has (an MNE Raw, an MNE Epochs, or a plain
ndarray) into exactly what one model expects, then runs the model. The contract
(sampling rate, units, channel order, window length, clipping) comes from the
registry entry, so the enforcement code lives here once rather than per model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..registry import ModelSpec

UNIT_TO_UV = {"uV": 1.0, "V": 1e6}


class AdapterError(RuntimeError):
    pass


def _as_array(
    data: Any, sfreq: float | None, ch_names: Sequence[str] | None
) -> tuple[np.ndarray, float, list[str], str]:
    """Return (array in microvolts, sfreq, channel names, input units) from MNE objects or ndarrays.

    MNE objects store EEG in volts; arrays are assumed to already be in microvolts
    unless ``units`` is passed to :meth:`BaseAdapter.prepare`.
    """
    cls = type(data).__name__
    if hasattr(data, "get_data") and hasattr(data, "info"):
        picks = None
        try:
            import mne  # noqa: F401

            picks = mne.pick_types(data.info, eeg=True, meg=False, eog=False, exclude="bads")
        except Exception:
            pass
        arr = data.get_data(picks=picks) if picks is not None else data.get_data()
        names = [data.ch_names[i] for i in picks] if picks is not None else list(data.ch_names)
        return np.asarray(arr, dtype=np.float32) * 1e6, float(data.info["sfreq"]), names, "V"
    arr = np.asarray(data, dtype=np.float32)
    if sfreq is None:
        raise AdapterError(
            f"sfreq is required when passing a {cls}; pass sfreq=... to prepare()/embed()/predict()."
        )
    if arr.ndim not in (2, 3):
        raise AdapterError(
            f"Expected (n_channels, n_times) or (n_epochs, n_channels, n_times); got shape {arr.shape}."
        )
    n_ch = arr.shape[-2]
    names = list(ch_names) if ch_names is not None else [f"ch{i}" for i in range(n_ch)]
    if len(names) != n_ch:
        raise AdapterError(f"{len(names)} channel names given for {n_ch} channels.")
    return arr, float(sfreq), names, "uV"


def resample(arr: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    """Resample along the last axis. Uses MNE when available, otherwise scipy, otherwise numpy FFT."""
    if abs(sfreq_in - sfreq_out) < 1e-6:
        return arr
    try:
        from mne.filter import resample as _mne_resample

        return _mne_resample(
            arr.astype(np.float64), up=sfreq_out, down=sfreq_in, axis=-1, verbose=False
        ).astype(np.float32)
    except ImportError:
        pass
    n_out = int(round(arr.shape[-1] * sfreq_out / sfreq_in))
    try:
        from scipy.signal import resample as _sp_resample

        return _sp_resample(arr, n_out, axis=-1).astype(np.float32)
    except ImportError:
        pass
    # numpy-only fallback: FFT resampling
    spec = np.fft.rfft(arr, axis=-1)
    n_in = arr.shape[-1]
    n_keep = min(spec.shape[-1], n_out // 2 + 1)
    out_spec = np.zeros(arr.shape[:-1] + (n_out // 2 + 1,), dtype=spec.dtype)
    out_spec[..., :n_keep] = spec[..., :n_keep]
    return (np.fft.irfft(out_spec, n=n_out, axis=-1) * (n_out / n_in)).astype(np.float32)


def normalize_ch_name(name: str) -> str:
    return name.strip().upper().replace("EEG", "").replace("-REF", "").replace("-LE", "").strip()


def select_channels(
    arr: np.ndarray, names: list[str], wanted: Sequence[str], strict: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Reorder/select channels to match ``wanted`` using case-insensitive 10-20 name matching."""
    targets = [normalize_ch_name(n) for n in wanted]
    if len(set(targets)) != len(targets):
        raise AdapterError("Ambiguous duplicate required channel names after normalization.")
    normalized = [normalize_ch_name(n) for n in names]
    if len(set(normalized)) != len(normalized):
        raise AdapterError("Ambiguous duplicate channel names after normalization.")
    lookup = {n: i for i, n in enumerate(normalized)}
    idx, missing = [], []
    for w in wanted:
        key = normalize_ch_name(w)
        if key in lookup:
            idx.append(lookup[key])
        else:
            missing.append(w)
    if missing and strict:
        raise AdapterError(f"Input is missing channels required by the model: {missing}")
    return arr[..., idx, :], [wanted[i] for i, w in enumerate(wanted) if normalize_ch_name(w) in lookup]


def window(arr: np.ndarray, sfreq: float, seconds: float, stride_seconds: float | None = None) -> np.ndarray:
    """Cut a continuous (n_channels, n_times) array into (n_windows, n_channels, n_samples)."""
    stride = seconds if stride_seconds is None else stride_seconds
    if not all(np.isfinite(x) and x > 0 for x in (sfreq, seconds, stride)):
        raise AdapterError("Sampling rate, window and stride must be finite and positive.")
    n = int(round(seconds * sfreq))
    step = int(round(stride * sfreq))
    if n < 1 or step < 1:
        raise AdapterError("Window and stride must each contain at least one sample.")
    if arr.shape[-1] < n:
        raise AdapterError(
            f"Recording is shorter ({arr.shape[-1]} samples) than one model window ({n} samples)."
        )
    starts = range(0, arr.shape[-1] - n + 1, step)
    return np.stack([arr[..., s : s + n] for s in starts], axis=0)


class BaseAdapter(ABC):
    """Subclass per model family. ``load_weights`` and ``forward`` are the only required methods."""

    def __init__(self, spec: ModelSpec, weights_dir: Path, device: str = "cpu", **kwargs: Any):
        self.spec = spec
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.kwargs = kwargs
        self.model: Any = None

    # ---- to implement -------------------------------------------------------
    @abstractmethod
    def load_weights(self) -> None:
        """Build ``self.model`` from ``self.weights_dir``."""

    @abstractmethod
    def forward(self, batch: np.ndarray) -> np.ndarray:
        """Run the model on (n_windows, n_channels, n_samples) float32 in the model's units."""

    # ---- shared --------------------------------------------------------------
    def validate_contract(self) -> None:
        """Fail before loading weights or transforming data when the recipe is unresolved."""
        inp = self.spec.input
        if not inp.contract_verified or inp.sfreq is None:
            raise AdapterError(
                f"'{self.spec.name}' input contract is not validated: "
                f"{inp.validation_notes or 'verify checkpoint preprocessing and montage first'}. "
                "See docs/MODEL_VALIDATION.md. Download availability is not inference readiness."
            )

    def prepare(
        self,
        data: Any,
        *,
        sfreq: float | None = None,
        ch_names: Sequence[str] | None = None,
        units: str | None = None,
        stride_seconds: float | None = None,
        strict_channels: bool = True,
    ) -> np.ndarray:
        """Enforce the registry input contract and return (n_windows, n_channels, n_samples)."""
        self.validate_contract()
        inp = self.spec.input
        arr, sf, names, in_units = _as_array(data, sfreq, ch_names)
        if units is not None and in_units == "uV":  # user told us what their array is in
            if units not in UNIT_TO_UV:
                raise AdapterError(f"units must be one of {sorted(UNIT_TO_UV)}")
            arr = arr * UNIT_TO_UV[units]
        if not np.isfinite(sf) or sf <= 0:
            raise AdapterError("sfreq must be finite and positive.")
        if not np.all(np.isfinite(arr)):
            raise AdapterError("Input contains non-finite values.")
        # arr is now in microvolts
        if inp.channels:
            arr, names = select_channels(arr, names, inp.channels, strict=strict_channels)
        elif inp.n_channels_max and arr.shape[-2] > inp.n_channels_max:
            raise AdapterError(f"Model accepts at most {inp.n_channels_max} channels; got {arr.shape[-2]}.")
        arr = resample(arr, sf, inp.sfreq)
        if inp.amplitude_clip_uv:
            arr = np.clip(arr, -inp.amplitude_clip_uv, inp.amplitude_clip_uv)
        if arr.ndim == 2:
            if inp.window_seconds:
                arr = window(arr, inp.sfreq, inp.window_seconds, stride_seconds)
            else:
                arr = arr[None]
        elif inp.window_seconds:
            n = int(round(inp.window_seconds * inp.sfreq))
            if arr.shape[-1] != n:
                raise AdapterError(
                    f"Epochs have {arr.shape[-1]} samples but the model window is {n} samples at {inp.sfreq} Hz."
                )
        self._ch_names = names
        return (arr / UNIT_TO_UV[inp.units]).astype(np.float32)  # convert uV -> model units

    def embed(self, data: Any, **kw: Any) -> np.ndarray:
        """Default: the adapter's forward output is treated as an embedding."""
        batch = self.prepare(data, **kw)
        if self.model is None:
            self.load_weights()
        return self.forward(batch)

    def predict(self, data: Any, **kw: Any) -> np.ndarray:
        """Default: same as embed. Override for models with a task head."""
        return self.embed(data, **kw)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.spec.name} ({'loaded' if self.model is not None else 'not loaded'})>"

"""Inverse modelling: noise covariance, inverse operator, source estimates and peak summaries."""

from __future__ import annotations

import mne
import numpy as np
from mne.minimum_norm import apply_inverse, make_inverse_operator

METHODS = ("MNE", "dSPM", "sLORETA", "eLORETA")


def compute_noise_covariance(
    epochs: mne.Epochs, tmin: float | None = None, tmax: float | None = 0.0, method: str = "empirical"
) -> mne.Covariance:
    """Noise covariance from the baseline window of the epochs."""
    return mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method=method, verbose="error")


def _with_average_ref_proj(inst):
    """MNE requires an average-reference *projector* for EEG inverse modelling."""
    has_proj = any("Average EEG reference" in p["desc"] for p in inst.info["projs"])
    if not has_proj:
        inst = inst.copy().set_eeg_reference("average", projection=True, verbose="error")
    return inst


def make_inverse(
    info_or_inst,
    fwd: mne.Forward,
    cov: mne.Covariance,
    loose: float | str = "auto",
    depth: float | None = 0.8,
):
    """Inverse operator. Accepts an Info, Raw, Epochs or Evoked; adds the average-reference
    projector if it is missing."""
    if isinstance(info_or_inst, mne.Info):
        # wrap the Info in an empty Evoked so the public set_eeg_reference API can add the projector
        info_or_inst = mne.EvokedArray(
            np.zeros((len(info_or_inst["ch_names"]), 1)), info_or_inst, verbose="error"
        )
    info = _with_average_ref_proj(info_or_inst).info
    return make_inverse_operator(info, fwd, cov, loose=loose, depth=depth, verbose="error")


def apply_inverse_evoked(evoked: mne.Evoked, inverse_operator, method: str = "dSPM", snr: float = 3.0):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    evoked = _with_average_ref_proj(evoked)
    lambda2 = 1.0 / snr**2
    stc = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None, verbose="error")
    return stc


def peak_source(stc, fwd: mne.Forward, tmin: float | None = None, tmax: float | None = None) -> dict:
    """Location and time of the strongest source, in head coordinates (mm)."""
    vert_idx, t_idx = stc.get_peak(tmin=tmin, tmax=tmax, vert_as_index=True, time_as_index=True)
    src = fwd["src"]
    if src.kind == "volume":
        rr = src[0]["rr"][src[0]["vertno"]]
        pos = rr[vert_idx]
        hemi = None
    else:
        n_lh = len(stc.vertices[0])
        if vert_idx < n_lh:
            hemi, vno = "lh", stc.vertices[0][vert_idx]
            pos = src[0]["rr"][vno]
        else:
            hemi, vno = "rh", stc.vertices[1][vert_idx - n_lh]
            pos = src[1]["rr"][vno]
    return {
        "time_s": float(stc.times[t_idx]),
        "value": float(stc.data[vert_idx, t_idx]),
        "position_mm": [round(float(v) * 1000, 1) for v in pos],
        "hemisphere": hemi,
        "n_sources": int(stc.data.shape[0]),
        "method": "unknown",
    }


def stc_summary(stc, times: tuple[float, float] | None = None) -> dict:
    data = stc.data if times is None else stc.copy().crop(*times).data
    return {
        "tmin_s": float(stc.tmin),
        "tmax_s": float(stc.times[-1]),
        "n_times": int(len(stc.times)),
        "n_sources": int(data.shape[0]),
        "max_abs": float(np.abs(data).max()),
        "mean_abs": float(np.abs(data).mean()),
    }

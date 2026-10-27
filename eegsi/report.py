"""HTML reports with mne.Report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import mne  # noqa: E402


def build_report(
    out_path: str | Path,
    title: str = "EEG source imaging report",
    raw: mne.io.BaseRaw | None = None,
    epochs: mne.Epochs | None = None,
    evoked: mne.Evoked | None = None,
    cov: mne.Covariance | None = None,
    fwd: mne.Forward | None = None,
    stc=None,
    extras: dict | None = None,
    log: list[str] | None = None,
) -> Path:
    """Assemble whatever has been computed so far into one self-contained HTML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep = mne.Report(title=title, verbose="error")
    if raw is not None:
        rep.add_raw(raw, title="Raw", psd=True, butterfly=False)
    if epochs is not None:
        rep.add_epochs(epochs, title="Epochs", psd=False)
    if evoked is not None:
        rep.add_evokeds(evoked, titles=["Evoked"], n_time_points=5)
        if cov is not None:
            rep.add_covariance(cov, info=evoked.info, title="Noise covariance")
    if fwd is not None:
        rep.add_forward(fwd, title="Forward solution")
    if stc is not None:
        fig = _stc_timecourse_figure(stc)
        rep.add_figure(fig, title="Source time courses", caption="Strongest 20 sources plus the mean.")
    if extras:
        pretty = html.escape(json.dumps(extras, indent=2, default=str))
        rep.add_html(f"<pre>{pretty}</pre>", title="Results")
    if log:
        items = "".join(f"<li>{html.escape(line)}</li>" for line in log)
        rep.add_html(f"<ol>{items}</ol>", title="Processing log")
    rep.save(out_path, overwrite=True, open_browser=False, verbose="error")
    return out_path


def _stc_timecourse_figure(stc):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 3.5))
    strongest = np.argsort(np.abs(stc.data).max(axis=1))[-20:]
    ax.plot(stc.times, stc.data[strongest].T, color="0.6", lw=0.6)
    ax.plot(stc.times, stc.data.mean(axis=0), color="C0", lw=2, label="mean")
    ax.axvline(0, color="k", lw=0.5)
    ax.set(xlabel="Time (s)", ylabel="Source amplitude")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

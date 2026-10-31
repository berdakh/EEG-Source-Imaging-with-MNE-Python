"""Forward modelling: a FreeSurfer-free spherical model, and fsaverage BEM when available."""

from __future__ import annotations

from pathlib import Path

import mne


def make_sphere_forward(info: mne.Info, pos: float = 0.015, mindist: float = 5.0) -> mne.Forward:
    """EEG forward solution with a 4-shell sphere head model and a volume source grid.

    Needs no MRI or FreeSurfer, so it runs anywhere. ``pos`` is the grid spacing in
    metres. Accuracy is lower than a BEM but the whole pipeline becomes testable.
    """
    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info, verbose="error")
    src = mne.setup_volume_source_space(sphere=sphere, pos=pos * 1000, mindist=mindist, verbose="error")
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose="error"
    )
    return fwd


def fsaverage_available(subjects_dir: str | Path | None = None) -> bool:
    try:
        subjects_dir = subjects_dir or mne.get_config("SUBJECTS_DIR")
        if not subjects_dir:
            return False
        return (Path(subjects_dir) / "fsaverage" / "bem").is_dir()
    except Exception:
        return False


def make_fsaverage_forward(
    info: mne.Info,
    spacing: str = "oct5",
    subjects_dir: str | Path | None = None,
    download: bool = True,
) -> mne.Forward:
    """EEG forward solution on the fsaverage template (3-layer BEM, cortical surface source space).

    Downloads fsaverage (~100 MB) on first use when ``download`` is true. The montage must be
    in head coordinates with fiducials, which is the case for MNE's standard montages.
    """
    if download:
        fs_dir = Path(mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose="error"))
        subjects_dir = fs_dir.parent
    subjects_dir = Path(subjects_dir or mne.get_config("SUBJECTS_DIR"))
    src = mne.setup_source_space(
        "fsaverage", spacing=spacing, subjects_dir=subjects_dir, add_dist=False, verbose="error"
    )
    bem_path = subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    fwd = mne.make_forward_solution(
        info, trans="fsaverage", src=src, bem=str(bem_path), eeg=True, meg=False, mindist=5.0, verbose="error"
    )
    return fwd


def describe_forward(fwd: mne.Forward) -> dict:
    n_sens, n_cols = fwd["sol"]["data"].shape
    kind = "surface" if fwd["src"].kind == "surface" else "volume"
    return {
        "n_sensors": int(n_sens),
        "n_sources": int(fwd["nsource"]),
        "source_space": kind,
        "orientation": "fixed" if fwd["source_ori"] == 1 else "free",
    }

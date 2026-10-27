"""Cluster-based permutation statistics at the sensor level."""

from __future__ import annotations

import mne
import numpy as np
from mne.stats import spatio_temporal_cluster_test


def sensor_cluster_test(
    epochs: mne.Epochs,
    condition_a: str,
    condition_b: str,
    n_permutations: int = 500,
    p_threshold: float = 0.05,
    seed: int = 0,
) -> dict:
    """Spatio-temporal cluster permutation test between two epoch conditions.

    Uses channel adjacency from the montage, an F-statistic with a parametric cluster
    forming threshold at ``p_threshold``, and returns a JSON-friendly summary.
    """
    for c in (condition_a, condition_b):
        if c not in epochs.event_id:
            raise ValueError(f"Unknown condition '{c}'. Available: {list(epochs.event_id)}")
    xa = epochs[condition_a].get_data(picks="eeg").transpose(0, 2, 1)  # (n_obs, n_times, n_ch)
    xb = epochs[condition_b].get_data(picks="eeg").transpose(0, 2, 1)
    adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, ch_type="eeg")
    from scipy import stats as sps

    dfn, dfd = 1, len(xa) + len(xb) - 2
    f_thresh = sps.f.ppf(1.0 - p_threshold, dfn, dfd)
    t_obs, clusters, p_values, _ = spatio_temporal_cluster_test(
        [xa, xb],
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=f_thresh,
        tail=1,
        seed=seed,
        out_type="mask",
        verbose="error",
    )
    ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]
    sig = []
    for mask, p in zip(clusters, p_values, strict=False):
        if p < p_threshold:
            t_idx, ch_idx = np.where(mask)
            sig.append(
                {
                    "p_value": float(p),
                    "tmin_s": float(epochs.times[t_idx.min()]),
                    "tmax_s": float(epochs.times[t_idx.max()]),
                    "channels": sorted({ch_names[i] for i in ch_idx}),
                    "n_points": int(mask.sum()),
                }
            )
    sig.sort(key=lambda d: d["p_value"])
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_trials": {condition_a: int(len(xa)), condition_b: int(len(xb))},
        "n_permutations": n_permutations,
        "cluster_threshold_F": float(f_thresh),
        "n_clusters": int(len(clusters)),
        "n_significant": len(sig),
        "significant_clusters": sig,
    }

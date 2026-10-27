# Legacy scripts

The original research scripts behind the 2018-2020 MRCP source-imaging work, kept
verbatim for reference and reproducibility of the published analyses. They have
hard-coded Windows paths, assume a FreeSurfer subjects directory, and target older
MNE and PySurfer APIs, so they will not run as-is.

The maintained, tested re-implementation of this functionality lives in the
[`eegsi`](../eegsi) package and is what the [`eegagent`](../eegagent) tools call.

| Script | What it does | Maintained equivalent |
| --- | --- | --- |
| `uh_prepfn.py`, `uh_prepoc.py`, `uh_EEGprep_main.py` | Notch/band-pass filtering, montage, epoching, outlier-based bad-channel detection, ICA | `eegsi.preprocessing` |
| `outlierdetection.py`, `badchannel-example.py` | Iterative variance outlier detector | `eegsi.preprocessing.find_bad_channels` |
| `uh_forward.py`, `uh_forwardbatch.py`, `plot_forward.py`, `read_source_space.py` | Source space, BEM, coregistration, forward solution | `eegsi.forward` |
| `uh_inverseBatch.py`, `uh_source_imaging.py`, `plot_mne_dspm_source_localization.py` | Noise covariance, inverse operator, dSPM/MNE, peak sources | `eegsi.inverse` |
| `stats_cluster_spatio_temporal.py`, `uh_stats_cluster_spatio_temporal_2samp.py`, `plot_sensor_permutation_test.py`, `background_statistics.py` | Cluster-based permutation statistics | `eegsi.stats` |
| `uh_report.py` | MNE HTML report | `eegsi.report` |
| `uhClass.py` | Class-based batch version of the above | all of `eegsi` |
| `mrcp_pipeline_v2.py`, `uh_pipeline.py`, `uh_paper_analysis*.py` | End-to-end pipelines used for the paper | `eegagent` tool chain |
| `batch*.py`, `move-file-shutil.py`, `uh_copydata.py`, `find_vmrk.py`, `load_mat_MRCP.py` | File-handling utilities for the lab's data layout | not ported |
| `MRCP_process.m` | MATLAB counterpart of the preprocessing | not ported |

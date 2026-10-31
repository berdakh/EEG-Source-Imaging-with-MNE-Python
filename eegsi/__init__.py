"""eegsi: EEG source imaging building blocks on top of MNE-Python.

Small, pure functions that take and return MNE objects, distilled from the legacy
research scripts. They are what the ``eegagent`` tools call, and are usable on
their own::

    from eegsi import data, preprocessing, forward, inverse, stats, report
"""

import mne

from . import data, forward, inverse, preprocessing, report, stats

mne.set_log_level("WARNING")

__all__ = ["data", "preprocessing", "forward", "inverse", "stats", "report"]
__version__ = "0.2.0"

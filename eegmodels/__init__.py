"""eegmodels: one registry, one downloader, one loader for open-weight EEG models.

Typical use::

    import eegmodels
    eegmodels.list_models()                 # names in the registry
    spec = eegmodels.info("steegformer-base")
    path = eegmodels.pull("steegformer-base")   # download weights into the cache
    model = eegmodels.load("steegformer-base")  # adapter with .embed(...) / .predict(...)
    z = model.embed(raw)                        # raw is an mne.io.Raw, Epochs or ndarray
"""

from .api import load
from .download import cache_dir, pull
from .registry import ModelSpec, get_spec, info, list_models

__all__ = [
    "ModelSpec",
    "get_spec",
    "list_models",
    "info",
    "pull",
    "cache_dir",
    "load",
]

__version__ = "0.1.0"

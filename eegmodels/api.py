"""High-level entry point: ``load(name)`` -> ready adapter."""

from __future__ import annotations

import os

from .adapters import BaseAdapter, get_adapter_class
from .download import model_dir, pull
from .registry import get_spec


def load(
    name: str,
    *,
    device: str = "cpu",
    cache: os.PathLike | str | None = None,
    download: bool = True,
    lazy: bool = False,
    **adapter_kwargs,
) -> BaseAdapter:
    """Resolve a registry entry, download its weights if needed, and return an adapter.

    ``lazy=True`` skips loading the network until the first ``embed``/``predict`` call.
    """
    spec = get_spec(name)
    kwargs = {**spec.adapter.kwargs, **adapter_kwargs}
    adapter = get_adapter_class(spec.adapter.cls)(spec, ".", device=device, **kwargs)
    adapter.validate_contract()
    adapter.weights_dir = model_dir(spec, cache)
    if download:
        adapter.weights_dir = pull(spec, cache=cache)
    if not lazy:
        adapter.load_weights()
    return adapter

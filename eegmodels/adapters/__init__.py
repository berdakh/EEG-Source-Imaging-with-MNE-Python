"""Adapter lookup. A registry entry names one of these keys under ``adapter.class``."""

from .base import AdapterError, BaseAdapter
from .braindecode import BraindecodeAdapter
from .planned import PlannedAdapter

ADAPTERS = {
    "braindecode": BraindecodeAdapter,
    "planned": PlannedAdapter,
}


def get_adapter_class(key: str):
    try:
        return ADAPTERS[key]
    except KeyError:
        raise AdapterError(f"Unknown adapter '{key}'. Known: {sorted(ADAPTERS)}") from None


__all__ = ["ADAPTERS", "AdapterError", "BaseAdapter", "get_adapter_class"]

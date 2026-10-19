"""Placeholder adapter for registry entries whose loader is not implemented yet."""

from __future__ import annotations

import numpy as np

from .base import AdapterError, BaseAdapter


class PlannedAdapter(BaseAdapter):
    def load_weights(self) -> None:
        raise AdapterError(
            f"'{self.spec.name}' is registered but its adapter is not implemented yet. "
            f"Code: {self.spec.references.get('code', 'n/a')}"
        )

    def forward(self, batch: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise AdapterError("not implemented")

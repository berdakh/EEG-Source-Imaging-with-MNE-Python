"""Feature-only adapter for the Braindecode foundation-model API.

Requires a validated input contract and a runtime exposing
``forward(..., return_features=True)``. Never falls back to classifier logits.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np

from .base import AdapterError, BaseAdapter


class BraindecodeAdapter(BaseAdapter):
    def load_weights(self) -> None:
        self.validate_contract()
        try:
            import braindecode.models as bdm
            import torch
        except ImportError as e:
            raise AdapterError(
                f"'{self.spec.name}' needs torch and a compatible braindecode: "
                "pip install 'eeg-source-imaging[braindecode]'"
            ) from e
        cls_name = self.kwargs.get("model_class")
        if not cls_name or not hasattr(bdm, cls_name):
            raise AdapterError(
                f"braindecode has no model class '{cls_name}' (check adapter.kwargs.model_class)."
            )
        cls = getattr(bdm, cls_name)
        if not hasattr(cls, "from_pretrained"):
            raise AdapterError(f"braindecode.models.{cls_name} has no from_pretrained().")
        if "return_features" not in inspect.signature(cls.forward).parameters:
            raise AdapterError(
                f"{cls_name} does not expose the required return_features API. "
                "Use a tested, pinned Braindecode version; logits are not embeddings."
            )
        self.model = cls.from_pretrained(str(self.weights_dir), **self.kwargs.get("model_kwargs", {}))
        self.model.to(self.device).eval()
        self._torch = torch

    def prepare(self, data, **kwargs) -> np.ndarray:
        # A fixed checkpoint montage cannot be made valid by silently dropping electrodes.
        if kwargs.get("strict_channels") is False and self.spec.input.channels:
            raise AdapterError("Fixed checkpoint channels require strict_channels=True.")
        return super().prepare(data, **kwargs)

    def forward(self, batch: np.ndarray) -> np.ndarray:
        """Return encoder features (possibly token-wise), never a task head's logits."""
        self.validate_contract()
        if self.model is None:
            raise AdapterError("Load model weights before calling forward().")
        expected = getattr(self.model, "n_chans", None)
        if expected is not None and batch.shape[-2] != expected:
            raise AdapterError(
                f"Checkpoint was constructed for {expected} channels, got {batch.shape[-2]}; "
                "validate the model construction and montage mapping."
            )
        if "return_features" not in inspect.signature(self.model.forward).parameters:
            raise AdapterError("Model lacks return_features; refusing to return classifier logits.")
        torch = self._torch
        x = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x, return_features=True)
        if not isinstance(out, Mapping) or "features" not in out or out["features"] is None:
            raise AdapterError("Expected the Braindecode feature mapping with a non-null 'features' tensor.")
        features = out["features"].detach().cpu().numpy()
        if features.ndim < 2 or features.shape[0] != batch.shape[0] or not np.isfinite(features).all():
            raise AdapterError("Model returned invalid encoder features (shape or non-finite values).")
        return features

    def predict(self, data, **kwargs) -> np.ndarray:
        raise AdapterError(
            "Task predictions are not validated for these backbone checkpoints. "
            "Use embed() for encoder features; a trained head and verified label mapping "
            "need a task-specific adapter."
        )

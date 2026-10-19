"""Adapter for models in braindecode's Hugging Face zoo (STEEGFormer, Signal-JEPA, MIRepNet, BrainBERT, MVPFormer).

braindecode >= 1.0 gives these models a ``from_pretrained`` classmethod backed by the
Hugging Face hub mixin, so the whole adapter is: import the class, load from the
cached snapshot, run it in eval mode.
"""

from __future__ import annotations

import numpy as np

from .base import AdapterError, BaseAdapter


class BraindecodeAdapter(BaseAdapter):
    def load_weights(self) -> None:
        try:
            import torch
            import braindecode.models as bdm
        except ImportError as e:
            raise AdapterError(
                f"'{self.spec.name}' needs torch and braindecode: pip install 'eegmodels[braindecode]'"
            ) from e
        cls_name = self.kwargs.get("model_class")
        if not cls_name or not hasattr(bdm, cls_name):
            raise AdapterError(f"braindecode has no model class '{cls_name}' (check adapter.kwargs.model_class).")
        cls = getattr(bdm, cls_name)
        if not hasattr(cls, "from_pretrained"):
            raise AdapterError(f"braindecode.models.{cls_name} has no from_pretrained(); upgrade braindecode.")
        self.model = cls.from_pretrained(str(self.weights_dir), **self.kwargs.get("model_kwargs", {}))
        self.model.to(self.device).eval()
        self._torch = torch

    def forward(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().cpu().numpy()

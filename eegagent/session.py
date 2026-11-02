"""Analysis session: the mutable state the tools read and write."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Session:
    """Everything one analysis has produced so far, plus a workspace directory for outputs."""

    workspace: Path = field(default_factory=lambda: Path("eegagent_workspace"))
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    raw: Any = None
    epochs: Any = None
    evoked: Any = None
    cov: Any = None
    fwd: Any = None
    inv: Any = None
    stc: Any = None
    bads: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.workspace = Path(self.workspace)
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return self.workspace / self.session_id

    def note(self, line: str) -> None:
        self.log.append(line)

    def describe(self) -> dict[str, Any]:
        """JSON-friendly snapshot of what exists, so the model can plan the next step."""
        d: dict[str, Any] = {"session_id": self.session_id, "workspace": str(self.dir)}
        if self.raw is not None:
            d["raw"] = {
                "n_channels": len(self.raw.ch_names),
                "sfreq": float(self.raw.info["sfreq"]),
                "duration_s": round(float(self.raw.times[-1]), 2),
                "highpass": float(self.raw.info["highpass"]),
                "lowpass": float(self.raw.info["lowpass"]),
                "bads": list(self.raw.info["bads"]),
                "has_montage": self.raw.get_montage() is not None,
            }
        if self.epochs is not None:
            d["epochs"] = {
                "n_epochs": len(self.epochs),
                "conditions": {k: int(len(self.epochs[k])) for k in self.epochs.event_id},
                "tmin": float(self.epochs.tmin),
                "tmax": float(self.epochs.tmax),
            }
        if self.evoked is not None:
            d["evoked"] = {"condition": self.evoked.comment, "nave": int(self.evoked.nave)}
        if self.fwd is not None:
            from eegsi.forward import describe_forward

            d["forward"] = describe_forward(self.fwd)
        if self.stc is not None:
            d["sources"] = self.results.get("sources")
        if self.bads:
            d["interpolated_bads"] = self.bads
        for k in ("cluster_test", "ica", "report", "embeddings"):
            if k in self.results:
                d[k] = self.results[k]
        d["steps_done"] = len(self.log)
        return d

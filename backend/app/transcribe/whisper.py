"""faster-whisper wrapper. Imported lazily so the app runs without the optional extra."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "faster-whisper is not installed. Run: uv sync --extra transcribe"
                ) from e
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self, path: Path, progress: Callable[[float], None] | None = None
    ) -> list[Segment]:
        model = self._load()
        segments, info = model.transcribe(str(path), vad_filter=True, beam_size=5)
        total = info.duration or 0.0
        out: list[Segment] = []
        for s in segments:
            out.append(Segment(start=s.start, end=s.end, text=s.text.strip()))
            if progress and total:
                progress(min(1.0, s.end / total))
        return out


def segments_to_text(segments: list[Segment]) -> str:
    return "\n".join(s.text for s in segments if s.text)

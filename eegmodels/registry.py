"""Model registry: one YAML file per model under ``eegmodels/registry/``.

The core code knows nothing about individual models. Everything model-specific
is data in the registry entry plus a small adapter class named by the entry.

Registry entry schema (all keys shown; optional ones are marked)::

    name: steegformer-base            # unique id used on the CLI and in load()
    family: steegformer               # groups variants of one architecture
    description: one line
    status: available | gated | planned
    source:
      type: hf | url                  # Hugging Face repo, or direct file URL(s)
      repo_id: braindecode/STEEGFormer-base   # hf only
      revision: main                  # hf only; pin a commit hash for reproducibility
      files:                          # url only: list of {url, sha256, filename}
        - url: https://...
          sha256: "..."
          filename: weights.pt
    license:
      spdx: MIT | Apache-2.0 | unknown | ...
      notes: free text shown before download   # optional
      research_only: false            # optional
    input:
      sfreq: 256                      # checkpoint Hz; null if unverified
      units: uV | V                   # amplitude units the model expects
      window_seconds: 4.0             # length of one input window
      channels: [Fp1, Fp2, ...]       # optional fixed channel order
      channel_vocab: 10-20            # optional: name of a channel vocabulary
      n_channels_max: 256             # optional
      amplitude_clip_uv: 100          # optional, only if required by checkpoint recipe
      channel_vocab_size: 145         # optional; NOT a maximum input channel count
      segment_seconds: 1.0           # optional tokenizer patch, NOT full context
      preprocessing: ["validated recipe description"]  # documentation, not executable
      contract_verified: false       # blocks inference until the contract is implemented/tested
      validation_notes: "remaining validation and primary evidence"
    adapter:
      class: braindecode              # key in eegmodels.adapters.ADAPTERS
      kwargs:                         # passed to the adapter constructor
        model_class: STEEGFormer
    references:                       # optional
      paper: https://...
      code: https://...
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_DIR = Path(__file__).parent / "registry"
VALID_STATUS = {"available", "gated", "planned"}
VALID_SOURCE_TYPES = {"hf", "url"}
VALID_UNITS = {"uV", "V"}


class RegistryError(ValueError):
    """Raised when a registry entry is malformed or a name is unknown."""


@dataclass
class SourceSpec:
    type: str
    repo_id: str | None = None
    revision: str = "main"
    files: list[dict[str, str]] = field(default_factory=list)


@dataclass
class LicenseSpec:
    spdx: str = "unknown"
    notes: str = ""
    research_only: bool = False


@dataclass
class InputSpec:
    sfreq: float | None
    units: str = "uV"
    window_seconds: float | None = None
    channels: list[str] | None = None
    channel_vocab: str | None = None
    n_channels_max: int | None = None
    amplitude_clip_uv: float | None = None
    channel_vocab_size: int | None = None
    segment_seconds: float | None = None
    preprocessing: list[str] = field(default_factory=list)
    validation_notes: str = ""
    contract_verified: bool = False


@dataclass
class AdapterSpec:
    cls: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    name: str
    family: str
    description: str
    status: str
    source: SourceSpec
    license: LicenseSpec
    input: InputSpec
    adapter: AdapterSpec
    references: dict[str, str] = field(default_factory=dict)
    path: Path | None = None  # registry file this came from

    @property
    def available(self) -> bool:
        return self.status in {"available", "gated"}

    def summary(self) -> str:
        lic = self.license.spdx + (" (research only)" if self.license.research_only else "")
        return f"{self.name:<22} {self.status:<10} {lic:<28} {self.description}"


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise RegistryError(f"{ctx}: missing required key '{key}'")
    return d[key]


def parse_spec(data: dict[str, Any], path: Path | None = None) -> ModelSpec:
    """Validate a raw dict (from YAML) and build a ModelSpec."""
    ctx = str(path) if path else "<dict>"
    name = _require(data, "name", ctx)
    status = _require(data, "status", ctx)
    if status not in VALID_STATUS:
        raise RegistryError(f"{ctx}: status must be one of {sorted(VALID_STATUS)}, got '{status}'")

    src = _require(data, "source", ctx)
    stype = _require(src, "type", f"{ctx}.source")
    if stype not in VALID_SOURCE_TYPES:
        raise RegistryError(f"{ctx}: source.type must be one of {sorted(VALID_SOURCE_TYPES)}")
    if stype == "hf" and not src.get("repo_id"):
        raise RegistryError(f"{ctx}: source.type 'hf' requires source.repo_id")
    if stype == "url":
        files = src.get("files") or []
        if not files:
            raise RegistryError(f"{ctx}: source.type 'url' requires a non-empty source.files list")
        for f in files:
            for k in ("url", "sha256", "filename"):
                if k not in f:
                    raise RegistryError(f"{ctx}: every source.files entry needs '{k}'")
    source = SourceSpec(
        type=stype,
        repo_id=src.get("repo_id"),
        revision=str(src.get("revision", "main")),
        files=list(src.get("files") or []),
    )

    lic_raw = data.get("license") or {}
    license_ = LicenseSpec(
        spdx=str(lic_raw.get("spdx", "unknown")),
        notes=str(lic_raw.get("notes", "")),
        research_only=bool(lic_raw.get("research_only", False)),
    )

    inp = _require(data, "input", ctx)
    units = inp.get("units", "uV")
    if units not in VALID_UNITS:
        raise RegistryError(f"{ctx}: input.units must be one of {sorted(VALID_UNITS)}")
    raw_sfreq = _require(inp, "sfreq", f"{ctx}.input")
    verified = inp.get("contract_verified", False)
    if not isinstance(verified, bool):
        raise RegistryError(f"{ctx}: input.contract_verified must be a boolean")
    if raw_sfreq is None and verified:
        raise RegistryError(f"{ctx}: verified input contract requires a sampling rate")
    numeric = ("sfreq", "window_seconds", "segment_seconds", "amplitude_clip_uv")
    for key in numeric:
        value = inp.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise RegistryError(f"{ctx}: input.{key} must be a finite positive number or null")
    for key in ("n_channels_max", "channel_vocab_size"):
        value = inp.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise RegistryError(f"{ctx}: input.{key} must be a positive integer or null")
    preprocessing = inp.get("preprocessing") or []
    if not isinstance(preprocessing, list) or not all(isinstance(x, str) for x in preprocessing):
        raise RegistryError(f"{ctx}: input.preprocessing must be a list of descriptions")
    channels = inp.get("channels")
    if channels is not None and (
        not isinstance(channels, list)
        or not channels
        or not all(isinstance(x, str) and x.strip() for x in channels)
        or len({x.strip().upper() for x in channels}) != len(channels)
    ):
        raise RegistryError(f"{ctx}: input.channels must contain unique nonempty channel names")
    input_ = InputSpec(
        sfreq=float(raw_sfreq) if raw_sfreq is not None else None,
        units=units,
        window_seconds=inp.get("window_seconds"),
        channels=inp.get("channels"),
        channel_vocab=inp.get("channel_vocab"),
        n_channels_max=inp.get("n_channels_max"),
        amplitude_clip_uv=inp.get("amplitude_clip_uv"),
        channel_vocab_size=inp.get("channel_vocab_size"),
        segment_seconds=inp.get("segment_seconds"),
        preprocessing=preprocessing,
        validation_notes=str(inp.get("validation_notes", "")),
        contract_verified=verified,
    )

    ad = _require(data, "adapter", ctx)
    adapter = AdapterSpec(cls=_require(ad, "class", f"{ctx}.adapter"), kwargs=dict(ad.get("kwargs") or {}))

    return ModelSpec(
        name=name,
        family=data.get("family", name),
        description=data.get("description", ""),
        status=status,
        source=source,
        license=license_,
        input=input_,
        adapter=adapter,
        references=dict(data.get("references") or {}),
        path=path,
    )


def registry_dirs() -> list[Path]:
    """Built-in registry dir plus any extra dirs from EEGMODELS_REGISTRY (os.pathsep-separated)."""
    dirs = [REGISTRY_DIR]
    extra = os.environ.get("EEGMODELS_REGISTRY")
    if extra:
        dirs += [Path(p) for p in extra.split(os.pathsep) if p]
    return dirs


def load_registry() -> dict[str, ModelSpec]:
    """Read every YAML file in the registry directories. Later dirs override earlier ones."""
    specs: dict[str, ModelSpec] = {}
    for d in registry_dirs():
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            spec = parse_spec(data, path)
            specs[spec.name] = spec
    return specs


def list_models(status: str | None = None) -> list[str]:
    """Names in the registry, optionally filtered by status."""
    specs = load_registry()
    names = [n for n, s in specs.items() if status is None or s.status == status]
    return sorted(names)


def get_spec(name: str) -> ModelSpec:
    specs = load_registry()
    if name not in specs:
        close = [n for n in specs if name.lower() in n.lower()]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise RegistryError(f"Unknown model '{name}'.{hint} Run `eegmodels list` to see all names.")
    return specs[name]


info = get_spec

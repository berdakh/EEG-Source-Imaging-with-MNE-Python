"""Tool definitions: a JSON schema plus a function that reads and writes the Session.

Each tool returns a JSON-serialisable dict. Errors are raised as ToolError and turned
into an error result by the agent loop, so the model can recover (for example by
loading a recording first).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eegsi import data, forward, inverse, preprocessing, report, stats

from .session import Session


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]
    required: list[str] = field(default_factory=list)

    def spec(self) -> dict[str, Any]:
        """Provider-neutral tool spec (JSON schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
                "additionalProperties": False,
            },
        }

    def __call__(self, session: Session, **kwargs: Any) -> dict[str, Any]:
        return self.fn(session, **kwargs)


TOOLS: dict[str, Tool] = {}


def tool(
    name: str, description: str, parameters: dict[str, Any] | None = None, required: list[str] | None = None
):
    def deco(fn):
        TOOLS[name] = Tool(name, description, parameters or {}, fn, required or [])
        return fn

    return deco


def _need(session: Session, attr: str, hint: str):
    if getattr(session, attr) is None:
        raise ToolError(f"No {attr} in the session yet. {hint}")


# ---------------------------------------------------------------------------- tools


@tool(
    "load_recording",
    "Load an EEG recording into the session. Accepts a file path (BrainVision .vhdr, .fif, .edf, "
    ".set, ...), 'simulated' for a built-in synthetic 32-channel recording with two event types "
    "(ids 1 and 2, a response peaking over left central electrodes at ~100 ms), or 'sample' for the "
    "MNE sample dataset (downloads on first use). Resets any previous results.",
    {"source": {"type": "string", "description": "File path, 'simulated' or 'sample'."}},
    ["source"],
)
def load_recording(session: Session, source: str) -> dict:
    if source == "simulated":
        raw = data.simulate_raw()
    elif source == "sample":
        raw = data.load_sample_raw()
    else:
        raw = data.load_recording(source)
    session.raw = raw
    session.epochs = session.evoked = session.cov = session.fwd = session.inv = session.stc = None
    session.bads = []
    session.results = {}
    events = data.find_events(raw)
    ids = sorted({int(e) for e in events[:, 2]}) if len(events) else []
    session.note(f"Loaded recording '{source}'")
    return {
        "n_channels": len(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]),
        "duration_s": round(float(raw.times[-1]), 2),
        "n_events": int(len(events)),
        "event_ids": ids,
        "has_montage": raw.get_montage() is not None,
        "channel_names": raw.ch_names[:40],
    }


@tool(
    "describe_state",
    "Summarise what the session currently contains (recording, epochs, forward model, sources...).",
)
def describe_state(session: Session) -> dict:
    return session.describe()


@tool(
    "filter_data",
    "Band-pass filter the continuous recording (zero-phase FIR) and optionally notch out line noise. "
    "Typical ERP/MRCP settings: l_freq 0.1-1, h_freq 30-40, notch 50 or 60.",
    {
        "l_freq": {"type": ["number", "null"], "description": "High-pass cutoff in Hz, null to skip."},
        "h_freq": {"type": ["number", "null"], "description": "Low-pass cutoff in Hz, null to skip."},
        "notch": {"type": ["number", "null"], "description": "Line-noise frequency (50 or 60) or null."},
    },
)
def filter_data(session: Session, l_freq=0.1, h_freq=40.0, notch=None) -> dict:
    _need(session, "raw", "Call load_recording first.")
    session.raw = preprocessing.apply_filters(session.raw, l_freq, h_freq, notch)
    session.epochs = session.evoked = session.stc = None
    session.note(f"Filtered {l_freq}-{h_freq} Hz" + (f", notch {notch} Hz" if notch else ""))
    return {
        "highpass": float(session.raw.info["highpass"]),
        "lowpass": float(session.raw.info["lowpass"]),
        "notch": notch,
    }


@tool(
    "detect_bad_channels",
    "Find channels whose variance is an outlier (iterative z-score, as in the original pipeline) and, "
    "by default, mark and interpolate them with spherical splines.",
    {
        "threshold": {
            "type": "number",
            "description": "z-score threshold on channel variance (default 3.0; the original pipeline used 2.0).",
        },
        "interpolate": {
            "type": "boolean",
            "description": "Interpolate the detected channels (default true).",
        },
    },
)
def detect_bad_channels(session: Session, threshold: float = 3.0, interpolate: bool = True) -> dict:
    _need(session, "raw", "Call load_recording first.")
    bads = preprocessing.find_bad_channels(session.raw, threshold=threshold)
    if bads and interpolate:
        session.raw = preprocessing.mark_and_interpolate(session.raw, bads)
        session.bads = sorted(set(session.bads) | set(bads))
        session.epochs = session.evoked = session.stc = None
    session.note(f"Bad channels {bads or 'none'}" + (" interpolated" if bads and interpolate else ""))
    return {"bad_channels": bads, "interpolated": bool(bads and interpolate)}


@tool(
    "make_epochs",
    "Cut the continuous data into epochs around the events. Conditions are named event_<id> unless "
    "event_names maps names to ids. Baseline-corrects using the pre-stimulus window and rejects "
    "epochs whose peak-to-peak amplitude exceeds reject_uv microvolts.",
    {
        "tmin": {
            "type": "number",
            "description": "Start of the epoch relative to the event, in seconds (e.g. -0.2).",
        },
        "tmax": {"type": "number", "description": "End of the epoch in seconds (e.g. 0.5)."},
        "reject_uv": {
            "type": ["number", "null"],
            "description": "Peak-to-peak rejection threshold in microvolts, null to disable.",
        },
        "event_names": {
            "type": ["object", "null"],
            "description": 'Optional mapping from condition name to event id, e.g. {"left": 1, "right": 2}.',
            "additionalProperties": {"type": "integer"},
        },
    },
    ["tmin", "tmax"],
)
def make_epochs(session: Session, tmin: float, tmax: float, reject_uv=150.0, event_names=None) -> dict:
    _need(session, "raw", "Call load_recording first.")
    event_id = {k: int(v) for k, v in event_names.items()} if event_names else None
    session.epochs = preprocessing.make_epochs(
        session.raw, tmin=tmin, tmax=tmax, event_id=event_id, reject_uv=reject_uv
    )
    session.evoked = session.cov = session.inv = session.stc = None
    counts = {k: int(len(session.epochs[k])) for k in session.epochs.event_id}
    session.note(f"Epoched {tmin}..{tmax} s: {counts}")
    return {
        "n_epochs": len(session.epochs),
        "conditions": counts,
        "n_rejected": int(len(session.epochs.drop_log) - len(session.epochs)),
    }


@tool(
    "set_average_reference",
    "Re-reference the epochs (or raw, if no epochs yet) to the common average.",
)
def set_average_reference(session: Session) -> dict:
    if session.epochs is not None:
        session.epochs = preprocessing.set_average_reference(session.epochs)
        target = "epochs"
    else:
        _need(session, "raw", "Call load_recording first.")
        session.raw = preprocessing.set_average_reference(session.raw)
        target = "raw"
    session.note(f"Average reference applied to {target}")
    return {"reference": "average", "applied_to": target}


@tool(
    "run_ica",
    "Fit ICA on the epochs and remove components that correlate with a frontal channel (EOG proxy). "
    "Use when blinks or eye movements are a concern.",
    {
        "n_components": {"type": "integer", "description": "Number of ICA components (default 15)."},
        "eog_channel": {
            "type": ["string", "null"],
            "description": "Frontal channel used as EOG proxy (default Fp1).",
        },
    },
)
def run_ica(session: Session, n_components: int = 15, eog_channel="Fp1") -> dict:
    _need(session, "epochs", "Call make_epochs first.")
    session.epochs, excluded = preprocessing.run_ica(
        session.epochs, n_components=n_components, eog_channel=eog_channel
    )
    session.evoked = session.stc = None
    session.results["ica"] = {"n_components": n_components, "excluded": excluded}
    session.note(f"ICA: removed components {excluded}")
    return session.results["ica"]


@tool(
    "compute_evoked",
    "Average the epochs of one condition into an evoked response and report its sensor-level peak.",
    {
        "condition": {
            "type": "string",
            "description": "Condition name from make_epochs (e.g. event_1). Omit to average all.",
        }
    },
)
def compute_evoked(session: Session, condition: str | None = None) -> dict:
    _need(session, "epochs", "Call make_epochs first.")
    ep = session.epochs if not condition else session.epochs[condition]
    if condition and condition not in session.epochs.event_id:
        raise ToolError(f"Unknown condition '{condition}'. Available: {list(session.epochs.event_id)}")
    session.evoked = ep.average()
    session.evoked.comment = condition or "all"
    tmin = max(0.0, float(session.evoked.times[0]))  # post-stimulus window only
    ch, t = session.evoked.get_peak(ch_type="eeg", tmin=tmin, mode="abs")
    amp = float(session.evoked.copy().crop(tmin, None).get_data(picks=ch).max()) * 1e6
    session.stc = None
    session.note(f"Evoked for '{condition or 'all'}' (nave={session.evoked.nave})")
    return {
        "condition": condition or "all",
        "nave": int(session.evoked.nave),
        "peak_channel": ch,
        "peak_time_s": float(t),
        "peak_amplitude_uv": round(amp, 2),
    }


@tool(
    "compute_forward",
    "Build the EEG forward model (leadfield). 'sphere' needs no MRI and works everywhere; "
    "'fsaverage' uses the FreeSurfer template cortex with a 3-layer BEM (downloads ~100 MB once).",
    {
        "model": {
            "type": "string",
            "enum": ["sphere", "fsaverage"],
            "description": "Head model (default sphere).",
        },
        "spacing_mm": {
            "type": "number",
            "description": "Volume grid spacing in mm for the sphere model (default 15).",
        },
    },
)
def compute_forward(session: Session, model: str = "sphere", spacing_mm: float = 15.0) -> dict:
    _need(session, "raw", "Call load_recording first.")
    info = session.epochs.info if session.epochs is not None else session.raw.info
    if model == "sphere":
        session.fwd = forward.make_sphere_forward(info, pos=spacing_mm / 1000.0)
    elif model == "fsaverage":
        session.fwd = forward.make_fsaverage_forward(info)
    else:
        raise ToolError("model must be 'sphere' or 'fsaverage'")
    session.inv = session.stc = None
    d = forward.describe_forward(session.fwd)
    d["model"] = model
    session.note(f"Forward model: {model} ({d['n_sources']} sources)")
    return d


@tool(
    "compute_sources",
    "Estimate cortical sources for the evoked response: noise covariance from the baseline, inverse "
    "operator, then MNE/dSPM/sLORETA/eLORETA. Reports the peak source location (head coordinates, mm) "
    "and time. Requires epochs, an evoked response and a forward model.",
    {
        "method": {
            "type": "string",
            "enum": ["MNE", "dSPM", "sLORETA", "eLORETA"],
            "description": "Inverse method (default dSPM).",
        },
        "snr": {"type": "number", "description": "Assumed SNR for regularisation (default 3)."},
        "peak_tmin": {
            "type": ["number", "null"],
            "description": "Restrict the peak search to times >= this (s).",
        },
        "peak_tmax": {
            "type": ["number", "null"],
            "description": "Restrict the peak search to times <= this (s).",
        },
    },
)
def compute_sources(
    session: Session, method: str = "dSPM", snr: float = 3.0, peak_tmin=None, peak_tmax=None
) -> dict:
    _need(session, "epochs", "Call make_epochs first.")
    _need(session, "evoked", "Call compute_evoked first.")
    _need(session, "fwd", "Call compute_forward first.")
    session.cov = inverse.compute_noise_covariance(session.epochs, tmax=0.0)
    session.inv = inverse.make_inverse(
        session.epochs,
        session.fwd,
        session.cov,
        loose=1.0 if session.fwd["src"].kind == "volume" else 0.2,
        depth=None,
    )
    session.stc = inverse.apply_inverse_evoked(session.evoked, session.inv, method=method, snr=snr)
    peak = inverse.peak_source(session.stc, session.fwd, tmin=peak_tmin, tmax=peak_tmax)
    peak["method"] = method
    out = {"method": method, "snr": snr, "peak": peak, "summary": inverse.stc_summary(session.stc)}
    session.results["sources"] = out
    session.note(f"Sources ({method}): peak at {peak['position_mm']} mm, t={peak['time_s']:.3f} s")
    return out


@tool(
    "cluster_test",
    "Non-parametric spatio-temporal cluster permutation test between two conditions at the sensor level. "
    "Returns the significant clusters with their time windows and channels.",
    {
        "condition_a": {"type": "string"},
        "condition_b": {"type": "string"},
        "n_permutations": {"type": "integer", "description": "Default 500."},
        "p_threshold": {
            "type": "number",
            "description": "Cluster-forming and reporting alpha (default 0.05).",
        },
    },
    ["condition_a", "condition_b"],
)
def cluster_test(
    session: Session, condition_a: str, condition_b: str, n_permutations: int = 500, p_threshold: float = 0.05
) -> dict:
    _need(session, "epochs", "Call make_epochs first.")
    res = stats.sensor_cluster_test(
        session.epochs, condition_a, condition_b, n_permutations=n_permutations, p_threshold=p_threshold
    )
    session.results["cluster_test"] = res
    session.note(f"Cluster test {condition_a} vs {condition_b}: {res['n_significant']} significant")
    return res


@tool(
    "generate_report",
    "Write an HTML report (MNE Report) with everything computed so far: raw PSD, epochs, evoked, "
    "covariance, forward model, source time courses, numeric results and the processing log. "
    "Returns the file path.",
    {"title": {"type": "string", "description": "Report title."}},
)
def generate_report(session: Session, title: str = "EEG source imaging report") -> dict:
    _need(session, "raw", "Nothing to report yet; load a recording first.")
    out = session.dir / "report.html"
    report.build_report(
        out,
        title=title,
        raw=session.raw,
        epochs=session.epochs,
        evoked=session.evoked,
        cov=session.cov,
        fwd=session.fwd,
        stc=session.stc,
        extras={k: v for k, v in session.results.items() if k != "report"},
        log=session.log,
    )
    session.results["report"] = {"path": str(out)}
    session.note(f"Report written to {out}")
    return {"path": str(out), "size_bytes": Path(out).stat().st_size}


# ---------------------------------------------------------------------------- helpers


def tool_specs() -> list[dict[str, Any]]:
    return [t.spec() for t in TOOLS.values()]


def run_tool(session: Session, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Execute a tool. Returns (result, is_error). Never raises for tool-level problems."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(TOOLS)}"}, True
    try:
        return TOOLS[name](session, **(args or {})), False
    except ToolError as e:
        return {"error": str(e)}, True
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}, True
    except Exception as e:  # analysis failures are information for the model, not crashes
        return {"error": f"{type(e).__name__}: {e}"}, True


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str)

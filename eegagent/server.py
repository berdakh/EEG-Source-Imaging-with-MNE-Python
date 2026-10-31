"""HTTP API around the agent: one Agent per session id, kept in memory.

uvicorn eegagent.server:app --port 8000

POST /sessions                     -> {"session_id": ...}
POST /sessions/{id}/chat {"message": "..."} -> answer + tool trace
GET  /sessions/{id}/state          -> session.describe()
GET  /sessions/{id}/report         -> the HTML report, if generated
GET  /tools                        -> tool specs
GET  /health
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import Agent
from .providers import get_provider
from .session import Session
from .tools import tool_specs

WORKSPACE = Path(os.environ.get("EEGAGENT_WORKSPACE", "eegagent_workspace"))

app = FastAPI(
    title="eegagent", version=__version__, description="LLM agent for EEG source imaging with MNE-Python."
)
_agents: dict[str, Agent] = {}
_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ToolEventOut(BaseModel):
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    is_error: bool
    seconds: float


class ChatResponse(BaseModel):
    session_id: str
    text: str
    trace: list[ToolEventOut]
    steps: int
    stop_reason: str
    usage: dict[str, int]
    state: dict[str, Any]


def _make_agent(session_id: str | None = None) -> Agent:
    provider = get_provider(os.environ.get("EEGAGENT_PROVIDER"), model=os.environ.get("EEGAGENT_MODEL"))
    session = Session(workspace=WORKSPACE, **({"session_id": session_id} if session_id else {}))
    return Agent(provider, session)


def _get(session_id: str) -> tuple[Agent, threading.Lock]:
    with _registry_lock:
        if session_id not in _agents:
            raise HTTPException(404, f"Unknown session '{session_id}'. POST /sessions first.")
        return _agents[session_id], _locks[session_id]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "provider": os.environ.get("EEGAGENT_PROVIDER", "anthropic"),
    }


@app.get("/tools")
def tools() -> list[dict[str, Any]]:
    return tool_specs()


@app.post("/sessions")
def create_session() -> dict[str, str]:
    agent = _make_agent()
    with _registry_lock:
        _agents[agent.session.session_id] = agent
        _locks[agent.session.session_id] = threading.Lock()
    return {
        "session_id": agent.session.session_id,
        "provider": agent.provider.name,
        "model": agent.provider.model,
    }


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, req: ChatRequest) -> ChatResponse:
    agent, lock = _get(session_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "This session is already processing a request.")
    try:
        res = agent.run(req.message)
    finally:
        lock.release()
    return ChatResponse(
        session_id=session_id,
        text=res.text,
        trace=[
            ToolEventOut(
                name=e.name, args=e.args, result=_jsonable(e.result), is_error=e.is_error, seconds=e.seconds
            )
            for e in res.trace
        ],
        steps=res.steps,
        stop_reason=res.stop_reason,
        usage=res.usage,
        state=_jsonable(agent.session.describe()),
    )


@app.get("/sessions/{session_id}/state")
def state(session_id: str) -> dict[str, Any]:
    agent, _ = _get(session_id)
    return _jsonable(agent.session.describe())


@app.get("/sessions/{session_id}/report")
def report(session_id: str):
    agent, _ = _get(session_id)
    path = (agent.session.results.get("report") or {}).get("path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "No report generated yet; ask the agent to generate one.")
    return FileResponse(path, media_type="text/html")


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    with _registry_lock:
        _agents.pop(session_id, None)
        _locks.pop(session_id, None)
    return {"deleted": session_id}


def _jsonable(obj: Any) -> Any:
    import json

    return json.loads(json.dumps(obj, default=str))

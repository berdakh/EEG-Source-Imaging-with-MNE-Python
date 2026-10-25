"""List and run pack workflows."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.audit import record
from app.config import Settings
from app.llm import OllamaClient
from app.workflows import Pack, RunContext, run_workflow

from .deps import get_conn, get_llm, get_pack, get_settings_dep

router = APIRouter(prefix="/workflows", tags=["workflows"])


class RunRequest(BaseModel):
    inputs: dict[str, Any]


@router.get("")
async def list_workflows(pack: Pack = Depends(get_pack)):
    return [
        {
            "id": w.id,
            "title": w.title,
            "description": w.description,
            "group": w.group,
            "inputs": [i.model_dump() for i in w.inputs],
            "outputs": [o.model_dump() for o in w.outputs],
        }
        for w in pack.workflows.values()
    ]


@router.post("/{workflow_id}/run")
async def run(
    workflow_id: str,
    req: RunRequest,
    pack: Pack = Depends(get_pack),
    llm: OllamaClient = Depends(get_llm),
    conn=Depends(get_conn),
    s: Settings = Depends(get_settings_dep),
):
    wf = pack.workflows.get(workflow_id)
    if wf is None:
        raise HTTPException(404, f"No workflow {workflow_id!r}")
    ctx = RunContext(pack=pack, llm=llm, conn=conn, outputs_dir=s.outputs_dir)
    try:
        result = await run_workflow(wf, req.inputs, ctx)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(
            503, f"Could not reach the local model server at {s.ollama_url}. Is Ollama running?"
        ) from e
    record(
        conn,
        "office",
        "workflow.run",
        {"workflow": workflow_id, "citations": [h.chunk_id for h in result.citations]},
    )
    return {
        "outputs": result.outputs,
        "citations": [h.__dict__ for h in result.citations],
        "files": [str(f) for f in result.files],
    }

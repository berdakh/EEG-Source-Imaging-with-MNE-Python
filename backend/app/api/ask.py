"""Ask the Office: a question over the library with citations.

Thin wrapper over the pack's `ask` workflow.
"""

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.audit import record
from app.llm import OllamaClient
from app.workflows import Pack, RunContext, run_workflow

from .deps import get_conn, get_llm, get_pack

router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    question: str


@router.post("")
async def ask(
    req: AskRequest,
    pack: Pack = Depends(get_pack),
    llm: OllamaClient = Depends(get_llm),
    conn=Depends(get_conn),
):
    wf = pack.workflows.get("ask")
    if wf is None:
        raise HTTPException(500, "This pack has no 'ask' workflow")
    ctx = RunContext(pack=pack, llm=llm, conn=conn)
    try:
        result = await run_workflow(wf, {"question": req.question}, ctx)
    except httpx.HTTPError as e:
        raise HTTPException(
            503, f"Could not reach the local model server at {llm.base_url}. Is Ollama running?"
        ) from e
    record(
        conn,
        "office",
        "ask",
        {"question": req.question, "citations": [h.chunk_id for h in result.citations]},
    )
    return {
        "answer": result.outputs.get("answer", ""),
        "citations": [h.__dict__ for h in result.citations],
    }

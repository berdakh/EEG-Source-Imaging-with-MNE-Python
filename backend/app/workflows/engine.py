"""Run a workflow: a sequence of steps sharing one state dict.

State keys are the workflow's input names and each step's id. Prompts see the
whole state, so a step can reference any earlier result by name.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.llm.prompts import PromptRenderer
from app.rag.search import Hit, hybrid_search
from app.render import write_docx, write_markdown

from .pack import Pack, Workflow, WorkflowStep


class LLM(Protocol):
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        options: dict | None = None,
    ) -> str: ...

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]: ...


class Transcribe(Protocol):
    def __call__(self, path: Path) -> str: ...


@dataclass
class RunContext:
    pack: Pack
    llm: LLM
    conn: sqlite3.Connection | None = None
    outputs_dir: Path | None = None
    transcribe: Transcribe | None = None
    citations: list[Hit] = field(default_factory=list)


@dataclass
class RunResult:
    state: dict[str, Any]
    outputs: dict[str, Any]
    citations: list[Hit]
    files: list[Path]


def _format_hits(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = f"{h.title}" + (f", page {h.page}" if h.page else "")
        blocks.append(f"[{i}] ({where})\n{h.text}")
    return "\n\n".join(blocks)


def _parse_json(text: str) -> Any:
    m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)


async def _run_step(
    step: WorkflowStep, state: dict[str, Any], ctx: RunContext, renderer: PromptRenderer
) -> Any:
    if step.type in {"generate", "extract"}:
        if not step.prompt:
            raise ValueError(f"step {step.id}: generate/extract needs a prompt")
        prompt = renderer.render(step.prompt, **state)
        system = renderer.render(step.system, **state) if step.system else None
        text = await ctx.llm.generate(prompt, system=system, model=step.model)
        return _parse_json(text) if step.type == "extract" else text

    if step.type == "retrieve":
        if ctx.conn is None:
            raise ValueError(f"step {step.id}: retrieve needs a database connection")
        query = str(state[step.source]) if step.source else str(state.get("question", ""))
        hits = await hybrid_search(ctx.conn, query, ctx.llm.embed, k=step.k)
        ctx.citations.extend(hits)
        return {"hits": [h.__dict__ for h in hits], "context": _format_hits(hits)}

    if step.type == "transcribe":
        if ctx.transcribe is None:
            raise ValueError(f"step {step.id}: no transcriber configured")
        audio = state[step.source or "audio"]
        return ctx.transcribe(Path(str(audio)))

    if step.type == "render":
        return state[step.source] if step.source else ""

    raise ValueError(f"unknown step type {step.type}")


async def run_workflow(workflow: Workflow, inputs: dict[str, Any], ctx: RunContext) -> RunResult:
    for inp in workflow.inputs:
        if inp.required and inp.name not in inputs:
            raise ValueError(f"missing required input {inp.name!r}")
    renderer = PromptRenderer(ctx.pack.prompts_dir)
    state: dict[str, Any] = dict(inputs)
    for step in workflow.steps:
        state[step.id] = await _run_step(step, state, ctx, renderer)

    outputs: dict[str, Any] = {}
    files: list[Path] = []
    for out in workflow.outputs:
        value = state.get(out.source)
        outputs[out.name] = value
        if ctx.outputs_dir and isinstance(value, str):
            base = ctx.outputs_dir / workflow.id / out.name
            if out.type == "docx":
                tpl = ctx.pack.templates_dir / out.template if out.template else None
                files.append(write_docx(value, base.with_suffix(".docx"), tpl))
            elif out.type == "markdown":
                files.append(write_markdown(value, base.with_suffix(".md")))
    return RunResult(state=state, outputs=outputs, citations=ctx.citations, files=files)

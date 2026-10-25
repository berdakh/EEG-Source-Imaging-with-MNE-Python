"""A pack is a folder of configuration that defines a vertical: prompts, workflows, templates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class WorkflowInput(BaseModel):
    name: str
    type: Literal["text", "document", "audio", "fields"] = "text"
    label: str | None = None
    required: bool = True
    description: str | None = None


class WorkflowStep(BaseModel):
    id: str
    type: Literal["generate", "retrieve", "transcribe", "extract", "render"]
    prompt: str | None = None  # generate/extract: template path relative to prompts/
    system: str | None = None  # generate/extract: optional system prompt template
    source: str | None = None  # retrieve: query key; transcribe: audio input key
    k: int = 6  # retrieve: number of hits
    schema_: dict | None = Field(default=None, alias="schema")  # extract: JSON schema hint
    template: str | None = None  # render: docx template relative to templates/
    model: str | None = None  # override model name


class WorkflowOutput(BaseModel):
    name: str
    source: str  # step id whose result is exported
    type: Literal["markdown", "docx", "json"] = "markdown"
    template: str | None = None


class Workflow(BaseModel):
    id: str
    title: str
    description: str = ""
    group: str = "General"
    inputs: list[WorkflowInput] = []
    steps: list[WorkflowStep]
    outputs: list[WorkflowOutput] = []


class NavItem(BaseModel):
    id: str
    label: str
    workflows: list[str] = []


class HardwareTier(BaseModel):
    min_memory_gb: int
    chat_model: str
    embed_model: str
    whisper_model: str = "small"


class PackManifest(BaseModel):
    name: str
    title: str
    description: str = ""
    version: str = "0.1.0"
    nav: list[NavItem] = []
    hardware_tiers: list[HardwareTier] = []


class Pack(BaseModel):
    root: Path
    manifest: PackManifest
    workflows: dict[str, Workflow]

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def templates_dir(self) -> Path:
        return self.root / "templates"


def load_pack(packs_dir: Path, name: str) -> Pack:
    root = packs_dir / name
    manifest_path = root / "pack.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No pack.yaml in {root}")
    manifest = PackManifest.model_validate(yaml.safe_load(manifest_path.read_text("utf-8")))
    workflows: dict[str, Workflow] = {}
    for wf_path in sorted((root / "workflows").glob("*.yaml")):
        wf = Workflow.model_validate(yaml.safe_load(wf_path.read_text("utf-8")))
        if wf.id in workflows:
            raise ValueError(f"Duplicate workflow id {wf.id!r} in {wf_path}")
        workflows[wf.id] = wf
    return Pack(root=root, manifest=manifest, workflows=workflows)

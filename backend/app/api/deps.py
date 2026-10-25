"""Shared request dependencies. Everything hangs off app.state, set up in main.lifespan."""

from fastapi import Request

from app.config import Settings
from app.jobs import JobQueue
from app.llm import OllamaClient
from app.workflows import Pack


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_conn(request: Request):
    return request.app.state.conn


def get_llm(request: Request) -> OllamaClient:
    return request.app.state.llm


def get_pack(request: Request) -> Pack:
    return request.app.state.pack


def get_jobs(request: Request) -> JobQueue:
    return request.app.state.jobs

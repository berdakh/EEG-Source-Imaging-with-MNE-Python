from fastapi import APIRouter

from . import ask, audit, health, jobs, library, settings, workflows

router = APIRouter(prefix="/api")
for mod in (health, settings, library, workflows, ask, jobs, audit):
    router.include_router(mod.router)

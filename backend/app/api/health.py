from fastapi import APIRouter, Depends

from app.config import Settings
from app.llm import OllamaClient
from app.workflows import Pack

from .deps import get_llm, get_pack, get_settings_dep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(
    s: Settings = Depends(get_settings_dep),
    llm: OllamaClient = Depends(get_llm),
    pack: Pack = Depends(get_pack),
):
    available = await llm.is_available()
    models = await llm.list_models() if available else []
    return {
        "ok": True,
        "pack": pack.manifest.name,
        "pack_title": pack.manifest.title,
        "ollama": {"available": available, "url": s.ollama_url, "models": models},
        "chat_model": s.chat_model,
        "embed_model": s.embed_model,
        "chat_model_installed": s.chat_model in models,
        "embed_model_installed": s.embed_model in models,
        "data_dir": str(s.data_dir),
    }

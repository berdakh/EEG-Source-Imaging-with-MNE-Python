from fastapi import APIRouter, Depends

from app.config import Settings
from app.workflows import Pack

from .deps import get_pack, get_settings_dep

router = APIRouter(tags=["settings"])


@router.get("/settings")
async def read_settings(s: Settings = Depends(get_settings_dep), pack: Pack = Depends(get_pack)):
    return {
        "data_dir": str(s.data_dir),
        "pack": s.pack,
        "chat_model": s.chat_model,
        "embed_model": s.embed_model,
        "whisper_model": s.whisper_model,
        "pin_enabled": bool(s.pin),
        "nav": [n.model_dump() for n in pack.manifest.nav],
        "hardware_tiers": [t.model_dump() for t in pack.manifest.hardware_tiers],
    }

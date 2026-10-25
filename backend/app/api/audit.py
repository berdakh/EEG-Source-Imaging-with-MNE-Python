from fastapi import APIRouter, Depends

from app.audit import recent

from .deps import get_conn

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("")
async def audit_log(limit: int = 100, conn=Depends(get_conn)):
    return recent(conn, limit)

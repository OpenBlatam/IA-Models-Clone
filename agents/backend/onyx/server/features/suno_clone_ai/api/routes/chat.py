"""
Endpoints para chat
"""

from typing import Dict, Any
from fastapi import APIRouter, Query, Depends

from ..dependencies import SongServiceDep

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


@router.get("/history/{user_id}")
async def get_chat_history(
    user_id: str,
    limit: int = Query(50, ge=1, le=100, description="Número máximo de mensajes"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """Obtiene el historial de chat de un usuario"""
    history = song_service.get_chat_history(user_id, limit=limit)
    return {"user_id": user_id, "history": history}


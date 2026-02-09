"""
Rutas para estadísticas de chats

Incluye: get_chat_stats, get_chat_stats_detailed
"""

from fastapi import APIRouter, Depends, Query

from ...dependencies import get_chat_service
from ...exceptions import ChatNotFoundError
from ...schemas import ChatStatsResponse
from ...services import ChatService
from ..decorators import handle_errors

router = APIRouter()


@router.get(
    "/chats/{chat_id}/stats",
    response_model=ChatStatsResponse,
    summary="Obtener estadísticas de un chat",
    description="Obtiene las estadísticas de engagement de un chat incluyendo su ranking"
)
@handle_errors
async def get_chat_stats(
    chat_id: str,
    detailed: bool = Query(False, description="Incluir estadísticas detalladas (upvotes, downvotes, engagement)"),
    service: ChatService = Depends(get_chat_service)
) -> ChatStatsResponse:
    chat = service.get_chat(chat_id)
    
    if not chat:
        raise ChatNotFoundError(chat_id)
    
    if detailed:
        stats = service.get_chat_stats_detailed(chat_id)
        return ChatStatsResponse(**stats)
    
    rank = service.get_chat_rank(chat_id)
    
    return ChatStatsResponse(
        chat_id=chat.id,
        vote_count=chat.vote_count,
        remix_count=chat.remix_count,
        view_count=chat.view_count,
        score=chat.score,
        rank=rank
    )


@router.get(
    "/chats/{chat_id}/stats/detailed",
    response_model=ChatStatsResponse,
    summary="Obtener estadísticas detalladas de un chat",
    description="Obtiene estadísticas detalladas incluyendo upvotes, downvotes y engagement rate"
)
@handle_errors
async def get_chat_stats_detailed(
    chat_id: str,
    service: ChatService = Depends(get_chat_service)
) -> ChatStatsResponse:
    stats = service.get_chat_stats_detailed(chat_id)
    return ChatStatsResponse(**stats)


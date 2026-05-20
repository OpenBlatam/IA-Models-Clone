"""
Rutas para búsqueda y descubrimiento de chats

Incluye: search_chats, get_top_chats, get_trending_chats, get_featured_chats
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from ...dependencies import get_chat_service, get_optional_user_id
from ...helpers import (
    build_chat_list_response,
    parse_tags_string,
    chats_to_responses,
    get_user_votes_for_chats
)
from ...schemas import (
    ChatListResponse,
    FeaturedChatsResponse,
    PublishedChatResponse,
    TrendingChatsResponse
)
from ...services import ChatService
from ...validators import validate_period
from ..cache import cache_response
from ..decorators import handle_errors

router = APIRouter()


@router.get(
    "/search",
    response_model=ChatListResponse,
    summary="Buscar chats",
    description="Busca chats por texto, tags, usuario, etc."
)
@handle_errors
async def search_chats(
    query: Optional[str] = Query(None, description="Texto de búsqueda"),
    tags: Optional[str] = Query(None, description="Tags separados por coma"),
    filter_user_id: Optional[str] = Query(None, alias="user_id", description="Filtrar por usuario"),
    sort_by: str = Query("score", description="Ordenar por"),
    order: str = Query("desc", description="Orden"),
    page: int = Query(1, ge=1, le=1000, description="Página"),
    page_size: int = Query(20, ge=1, le=100, description="Tamaño de página"),
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    service: ChatService = Depends(get_chat_service)
) -> ChatListResponse:
    tags_list = parse_tags_string(tags)
    
    chats, total = service.search_chats(
        query=query,
        tags=tags_list,
        user_id=filter_user_id,
        sort_by=sort_by,
        order=order,
        page=page,
        page_size=page_size
    )
    
    return build_chat_list_response(
        chats=chats,
        total=total,
        page=page,
        page_size=page_size,
        current_user_id=current_user_id,
        service=service
    )


@router.get(
    "/top",
    response_model=List[PublishedChatResponse],
    summary="Obtener chats más populares",
    description="Obtiene los chats con mayor score (ranking)"
)
@cache_response(ttl=60.0)
@handle_errors
async def get_top_chats(
    limit: int = Query(20, ge=1, le=100, description="Límite de resultados"),
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    service: ChatService = Depends(get_chat_service)
) -> List[PublishedChatResponse]:
    chats = service.get_top_chats(limit)
    user_votes = get_user_votes_for_chats(chats, current_user_id, service)
    
    return chats_to_responses(chats, current_user_id, user_votes)


@router.get(
    "/trending",
    response_model=TrendingChatsResponse,
    summary="Obtener chats trending",
    description="Obtiene los chats trending en diferentes períodos de tiempo"
)
@handle_errors
async def get_trending_chats(
    period: str = Query("day", description="Período: hour, day, week, month"),
    limit: int = Query(20, ge=1, le=100, description="Límite de resultados"),
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    service: ChatService = Depends(get_chat_service)
) -> TrendingChatsResponse:
    period = validate_period(period)
    
    chats = service.get_trending_chats(period, limit)
    user_votes = get_user_votes_for_chats(chats, current_user_id, service)
    
    return TrendingChatsResponse(
        period=period,
        chats=chats_to_responses(chats, current_user_id, user_votes)
    )


@router.get(
    "/featured",
    response_model=FeaturedChatsResponse,
    summary="Obtener chats destacados",
    description="Obtiene todos los chats destacados (featured) ordenados por score"
)
@cache_response(ttl=120.0)
@handle_errors
async def get_featured_chats(
    limit: int = Query(50, ge=1, le=100, description="Límite de resultados"),
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    service: ChatService = Depends(get_chat_service)
) -> FeaturedChatsResponse:
    chats, user_votes = service.get_featured_chats(limit, current_user_id)
    
    chat_responses = chats_to_responses(chats, current_user_id, user_votes)
    
    return FeaturedChatsResponse(
        chats=chat_responses,
        total=len(chat_responses)
    )


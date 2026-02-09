"""
Rutas para operaciones de chats

Incluye: publish, list, get, update, delete, feature
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, status

from ...dependencies import (
    get_chat_service,
    get_optional_user_id,
    get_user_id
)
from ...exceptions import ChatNotFoundError, InvalidChatError
from ...helpers import chat_to_response, build_chat_list_response
from ...schemas import (
    ChatListResponse,
    PublishChatRequest,
    PublishedChatResponse,
    UpdateChatRequest
)
from ...services import ChatService
from ..cache import cache_response
from ..decorators import handle_errors

router = APIRouter()


@router.post(
    "/publish",
    response_model=PublishedChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Publicar un chat",
    description="Publica un chat en la comunidad para que otros usuarios puedan verlo y remixarlo"
)
@handle_errors
async def publish_chat(
    request: PublishChatRequest,
    user_id: str = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service)
) -> PublishedChatResponse:
    chat = service.publish_chat(
        user_id=user_id,
        title=request.title,
        chat_content=request.chat_content,
        description=request.description,
        tags=request.tags,
        is_public=request.is_public
    )
    return chat_to_response(chat)


@router.get(
    "/chats",
    response_model=ChatListResponse,
    summary="Listar chats",
    description="Obtiene una lista de chats con paginación y ordenamiento por score"
)
@cache_response(ttl=30.0)
@handle_errors
async def list_chats(
    page: int = Query(1, ge=1, le=1000, description="Página"),
    page_size: int = Query(20, ge=1, le=100, description="Tamaño de página"),
    sort_by: str = Query("score", description="Ordenar por: score, created_at, vote_count, remix_count"),
    order: str = Query("desc", description="Orden: asc o desc"),
    filter_user_id: Optional[str] = Query(None, alias="user_id", description="Filtrar por usuario"),
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    service: ChatService = Depends(get_chat_service)
) -> ChatListResponse:
    chats, total = service.search_chats(
        sort_by=sort_by,
        order=order,
        page=page,
        page_size=page_size,
        user_id=filter_user_id
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
    "/chats/{chat_id}",
    response_model=PublishedChatResponse,
    summary="Obtener un chat",
    description="Obtiene los detalles de un chat específico y registra una visualización"
)
@handle_errors
async def get_chat(
    chat_id: str,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    service: ChatService = Depends(get_chat_service)
) -> PublishedChatResponse:
    chat = service.get_chat(chat_id, current_user_id)
    
    if not chat:
        raise ChatNotFoundError(chat_id)
    
    user_vote = None
    if current_user_id:
        user_vote = service.get_user_vote(chat_id, current_user_id)
    
    return chat_to_response(chat, current_user_id, user_vote)


@router.put(
    "/chats/{chat_id}",
    response_model=PublishedChatResponse,
    summary="Actualizar un chat",
    description="Actualiza los campos de un chat existente. Solo el propietario puede actualizar su chat."
)
@handle_errors
async def update_chat(
    chat_id: str,
    request: UpdateChatRequest,
    user_id: str = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service)
) -> PublishedChatResponse:
    chat = service.update_chat(
        chat_id=chat_id,
        user_id=user_id,
        title=request.title,
        description=request.description,
        tags=request.tags,
        is_public=request.is_public
    )
    
    user_vote = service.get_user_vote(chat_id, user_id)
    return chat_to_response(chat, user_id, user_vote)


@router.delete(
    "/chats/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Eliminar un chat",
    description="Elimina un chat. Solo el propietario puede eliminar su chat."
)
@handle_errors
async def delete_chat(
    chat_id: str,
    user_id: str = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service)
) -> None:
    service.delete_chat(chat_id, user_id)


@router.post(
    "/chats/{chat_id}/feature",
    response_model=PublishedChatResponse,
    summary="Destacar un chat",
    description="Marca un chat como destacado (featured). Requiere permisos de administrador."
)
@handle_errors
async def feature_chat(
    chat_id: str,
    featured: bool = Query(True, description="True para destacar, False para quitar destacado"),
    service: ChatService = Depends(get_chat_service)
) -> PublishedChatResponse:
    chat = service.feature_chat(chat_id, featured)
    return chat_to_response(chat)


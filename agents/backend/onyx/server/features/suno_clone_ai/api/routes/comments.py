"""
Endpoints para sistema de comentarios

Este módulo proporciona funcionalidades para comentar canciones:
- Agregar comentarios
- Obtener comentarios con paginación
- Eliminar comentarios
- Estadísticas de comentarios

Características:
- Validación de longitud
- Ordenamiento por fecha
- Paginación eficiente
- Tracking de comentarios
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, status

from ..dependencies import SongServiceDep
from ..validators import validate_song_id, ensure_song_exists

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["comments"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        404: {"description": "Not found - Canción no encontrada"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/{song_id}/comments",
    summary="Agregar comentario",
    description="Agrega un comentario a una canción"
)
async def add_comment(
    song_id: str,
    comment: str = Query(..., min_length=1, max_length=500, description="Texto del comentario"),
    user_id: str = Query(..., min_length=1, max_length=100, description="ID del usuario"),
    parent_comment_id: Optional[str] = Query(None, description="ID del comentario padre (para respuestas)"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Agrega un comentario a una canción.
    
    Soporta comentarios directos y respuestas a otros comentarios (threading).
    
    Args:
        song_id: ID único de la canción (UUID)
        comment: Texto del comentario (1-500 caracteres)
        user_id: ID del usuario que comenta
        parent_comment_id: ID del comentario padre (opcional, para respuestas)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación con información del comentario
    
    Raises:
        HTTPException 400: Si el formato es inválido
        HTTPException 404: Si la canción o comentario padre no existe
    
    Example:
        ```
        POST /suno/songs/123e4567-e89b-12d3-a456-426614174000/comments?comment=Great song!&user_id=user123
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        # Validar comentario padre si se proporciona
        if parent_comment_id:
            metadata = song.get("metadata", {})
            comments = metadata.get("comments", [])
            parent_exists = any(c.get("id") == parent_comment_id for c in comments)
            if not parent_exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Parent comment {parent_comment_id} not found"
                )
        
        comment_id = str(uuid.uuid4())
        new_comment = {
            "id": comment_id,
            "user_id": user_id,
            "comment": comment.strip(),
            "created_at": datetime.now().isoformat(),
            "parent_comment_id": parent_comment_id,
            "replies_count": 0,
            "likes_count": 0
        }
        
        metadata = song.get("metadata", {})
        comments = metadata.get("comments", [])
        comments.append(new_comment)
        
        # Incrementar contador de respuestas del comentario padre
        if parent_comment_id:
            for c in comments:
                if c.get("id") == parent_comment_id:
                    c["replies_count"] = c.get("replies_count", 0) + 1
                    break
        
        metadata["comments"] = comments
        metadata["comment_count"] = len(comments)
        
        song_service.save_song(
            song_id=song_id,
            user_id=song.get("user_id"),
            prompt=song.get("prompt", ""),
            file_path=song.get("file_path", ""),
            metadata=metadata
        )
        
        logger.info(f"Comment {comment_id} added to song {song_id} by user {user_id}")
        
        return {
            "message": "Comment added successfully",
            "comment_id": comment_id,
            "song_id": song_id,
            "parent_comment_id": parent_comment_id,
            "comment": new_comment
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding comment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding comment: {str(e)}"
        )


@router.get(
    "/{song_id}/comments",
    summary="Obtener comentarios",
    description="Obtiene los comentarios de una canción con paginación"
)
async def get_comments(
    song_id: str,
    limit: int = Query(50, ge=1, le=100, description="Número máximo de comentarios"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    sort_by: Optional[str] = Query("created_at", description="Ordenar por: created_at, likes"),
    include_replies: bool = Query(True, description="Incluir respuestas a comentarios"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene los comentarios de una canción.
    
    Soporta ordenamiento y puede incluir/excluir respuestas (threading).
    
    Args:
        song_id: ID único de la canción (UUID)
        limit: Número máximo de comentarios
        offset: Offset para paginación
        sort_by: Campo para ordenar (created_at, likes)
        include_replies: Incluir respuestas a comentarios
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de comentarios con paginación
    
    Example:
        ```
        GET /suno/songs/123e4567-e89b-12d3-a456-426614174000/comments?limit=20&sort_by=likes
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        all_comments = song.get("metadata", {}).get("comments", [])
        
        # Filtrar comentarios principales o incluir respuestas
        if include_replies:
            comments_to_show = all_comments
        else:
            comments_to_show = [c for c in all_comments if not c.get("parent_comment_id")]
        
        # Ordenar
        if sort_by == "likes":
            comments_to_show.sort(
                key=lambda x: x.get("likes_count", 0),
                reverse=True
            )
        else:  # created_at (default)
            comments_to_show.sort(
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )
        
        total = len(comments_to_show)
        paginated_comments = comments_to_show[offset:offset + limit]
        has_more = (offset + limit) < total
        
        # Agrupar respuestas si se incluyen
        if include_replies:
            comments_with_replies = []
            for comment in paginated_comments:
                if not comment.get("parent_comment_id"):
                    # Es un comentario principal, buscar sus respuestas
                    replies = [
                        c for c in all_comments
                        if c.get("parent_comment_id") == comment.get("id")
                    ]
                    comment_with_replies = {**comment, "replies": replies}
                    comments_with_replies.append(comment_with_replies)
                else:
                    # Es una respuesta, solo incluir si el padre está en la página
                    parent_in_page = any(
                        c.get("id") == comment.get("parent_comment_id")
                        for c in paginated_comments
                    )
                    if parent_in_page:
                        comments_with_replies.append(comment)
            
            paginated_comments = comments_with_replies
        
        return {
            "song_id": song_id,
            "comments": paginated_comments,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "sort_by": sort_by,
            "include_replies": include_replies
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comments: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting comments: {str(e)}"
        )


@router.delete("/{song_id}/comments/{comment_id}")
async def delete_comment(
    song_id: str,
    comment_id: str,
    user_id: str = Query(..., description="ID del usuario (debe ser el autor)"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Elimina un comentario de una canción.
    
    Solo el autor del comentario puede eliminarlo.
    
    Args:
        song_id: ID único de la canción (UUID)
        comment_id: ID del comentario a eliminar
        user_id: ID del usuario (debe ser el autor)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación
    
    Raises:
        HTTPException 403: Si el usuario no es el autor
        HTTPException 404: Si el comentario no existe
    
    Example:
        ```
        DELETE /suno/songs/123e4567-e89b-12d3-a456-426614174000/comments/comment456?user_id=user123
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        metadata = song.get("metadata", {})
        comments = metadata.get("comments", [])
        
        # Buscar comentario
        comment_to_delete = None
        for i, comment in enumerate(comments):
            if comment.get("id") == comment_id:
                comment_to_delete = comment
                comment_index = i
                break
        
        if not comment_to_delete:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Comment {comment_id} not found"
            )
        
        # Verificar autoría
        if comment_to_delete.get("user_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete your own comments"
            )
        
        # Eliminar comentario
        parent_id = comment_to_delete.get("parent_comment_id")
        comments.pop(comment_index)
        
        # Decrementar contador de respuestas del padre si existe
        if parent_id:
            for comment in comments:
                if comment.get("id") == parent_id:
                    comment["replies_count"] = max(0, comment.get("replies_count", 0) - 1)
                    break
        
        metadata["comments"] = comments
        metadata["comment_count"] = len(comments)
        
        song_service.save_song(
            song_id=song_id,
            user_id=song.get("user_id"),
            prompt=song.get("prompt", ""),
            file_path=song.get("file_path", ""),
            metadata=metadata
        )
        
        logger.info(f"Comment {comment_id} deleted from song {song_id} by user {user_id}")
        
        return {
            "message": "Comment deleted successfully",
            "comment_id": comment_id,
            "song_id": song_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting comment: {str(e)}"
        )


@router.post("/{song_id}/comments/{comment_id}/like")
async def like_comment(
    song_id: str,
    comment_id: str,
    user_id: str = Query(..., description="ID del usuario"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Da like a un comentario.
    
    Args:
        song_id: ID único de la canción (UUID)
        comment_id: ID del comentario
        user_id: ID del usuario que da like
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Mensaje de confirmación con contador actualizado
    
    Example:
        ```
        POST /suno/songs/123e4567-e89b-12d3-a456-426614174000/comments/comment456/like?user_id=user123
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        metadata = song.get("metadata", {})
        comments = metadata.get("comments", [])
        
        # Buscar comentario
        comment_found = False
        for comment in comments:
            if comment.get("id") == comment_id:
                comment_found = True
                likes = comment.get("likes", [])
                
                if user_id not in likes:
                    likes.append(user_id)
                    comment["likes"] = likes
                    comment["likes_count"] = len(likes)
                
                break
        
        if not comment_found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Comment {comment_id} not found"
            )
        
        metadata["comments"] = comments
        
        song_service.save_song(
            song_id=song_id,
            user_id=song.get("user_id"),
            prompt=song.get("prompt", ""),
            file_path=song.get("file_path", ""),
            metadata=metadata
        )
        
        return {
            "message": "Comment liked",
            "comment_id": comment_id,
            "likes_count": comment.get("likes_count", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error liking comment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error liking comment: {str(e)}"
        )


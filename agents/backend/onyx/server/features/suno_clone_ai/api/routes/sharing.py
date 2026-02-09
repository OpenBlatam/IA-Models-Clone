"""
Endpoints para compartición segura de canciones

Este módulo proporciona funcionalidades para compartir canciones de forma segura:
- Crear enlaces de compartición con expiración
- Validar tokens de compartición
- Gestionar múltiples enlaces por canción
- Estadísticas de compartición

Características:
- Tokens únicos y seguros (UUID)
- Expiración configurable (1h-7d)
- Validación de expiración
- Tracking de comparticiones
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, status, Depends

from ..dependencies import SongServiceDep
from ..validators import validate_song_id, ensure_song_exists

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/songs",
    tags=["sharing"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        404: {"description": "Not found - Canción o token no encontrado"},
        410: {"description": "Gone - Enlace expirado"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/{song_id}/share",
    summary="Crear enlace de compartición",
    description="Crea un enlace de compartición seguro con expiración para una canción"
)
async def create_share_link(
    song_id: str,
    expires_in: Optional[int] = Query(86400, ge=3600, le=604800, description="Tiempo de expiración en segundos (1h-7d)"),
    max_uses: Optional[int] = Query(None, ge=1, description="Número máximo de usos (opcional)"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Crea un enlace de compartición seguro para una canción.
    
    El enlace tiene un token único y puede tener:
    - Expiración temporal (1 hora a 7 días)
    - Límite de usos (opcional)
    
    Args:
        song_id: ID único de la canción (UUID)
        expires_in: Tiempo de expiración en segundos (3600-604800)
        max_uses: Número máximo de veces que se puede usar el enlace (opcional)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Información del enlace de compartición creado
    
    Raises:
        HTTPException 400: Si el song_id es inválido
        HTTPException 404: Si la canción no existe
    
    Example:
        ```
        POST /suno/songs/123e4567-e89b-12d3-a456-426614174000/share?expires_in=86400&max_uses=10
        ```
    """
    try:
        validate_song_id(song_id)
        ensure_song_exists(song_service.get_song(song_id), song_id)
        
        share_token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        share_link = f"/suno/songs/shared/{share_token}"
        
        # Guardar token en metadata
        song = song_service.get_song(song_id)
        metadata = song.get("metadata", {})
        shares = metadata.get("shares", [])
        
        share_data = {
            "token": share_token,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "expires_in": expires_in,
            "max_uses": max_uses,
            "uses_count": 0,
            "is_active": True
        }
        
        shares.append(share_data)
        metadata["shares"] = shares
        metadata["share_count"] = len(shares)
        
        song_service.save_song(
            song_id=song_id,
            user_id=song.get("user_id"),
            prompt=song.get("prompt", ""),
            file_path=song.get("file_path", ""),
            metadata=metadata
        )
        
        logger.info(f"Share link created for song {song_id} (expires: {expires_at.isoformat()})")
        
        return {
            "message": "Share link created",
            "song_id": song_id,
            "share_link": share_link,
            "share_token": share_token,
            "expires_at": expires_at.isoformat(),
            "expires_in": expires_in,
            "max_uses": max_uses,
            "uses_remaining": max_uses if max_uses else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating share link: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating share link: {str(e)}"
        )


@router.get(
    "/shared/{share_token}",
    summary="Obtener canción compartida",
    description="Obtiene una canción compartida mediante un token válido"
)
async def get_shared_song(
    share_token: str,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene una canción compartida mediante un token.
    
    Valida:
    - Que el token exista
    - Que no haya expirado
    - Que no haya excedido el límite de usos (si aplica)
    
    Args:
        share_token: Token de compartición (UUID)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Información de la canción compartida
    
    Raises:
        HTTPException 404: Si el token no existe
        HTTPException 410: Si el enlace ha expirado o excedido usos
    
    Example:
        ```
        GET /suno/songs/shared/abc123def456
        ```
    """
    try:
        all_songs = song_service.list_songs(limit=10000, offset=0)
        
        for song in all_songs:
            shares = song.get("metadata", {}).get("shares", [])
            for share in shares:
                if share.get("token") == share_token:
                    # Verificar si está activo
                    if not share.get("is_active", True):
                        raise HTTPException(
                            status_code=status.HTTP_410_GONE,
                            detail="Share link has been deactivated"
                        )
                    
                    # Verificar expiración
                    expires_at_str = share.get("expires_at")
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if datetime.now() > expires_at:
                            raise HTTPException(
                                status_code=status.HTTP_410_GONE,
                                detail="Share link has expired"
                            )
                    
                    # Verificar límite de usos
                    max_uses = share.get("max_uses")
                    uses_count = share.get("uses_count", 0)
                    if max_uses and uses_count >= max_uses:
                        raise HTTPException(
                            status_code=status.HTTP_410_GONE,
                            detail=f"Share link has reached maximum uses ({max_uses})"
                        )
                    
                    # Incrementar contador de usos
                    share["uses_count"] = uses_count + 1
                    song.get("metadata", {})["shares"] = shares
                    
                    # Guardar actualización
                    song_service.save_song(
                        song_id=song.get("song_id"),
                        user_id=song.get("user_id"),
                        prompt=song.get("prompt", ""),
                        file_path=song.get("file_path", ""),
                        metadata=song.get("metadata", {})
                    )
                    
                    logger.info(f"Share token {share_token} used (uses: {uses_count + 1})")
                    
                    return {
                        "song": song,
                        "shared": True,
                        "expires_at": share.get("expires_at"),
                        "uses_count": uses_count + 1,
                        "max_uses": max_uses,
                        "uses_remaining": (max_uses - uses_count - 1) if max_uses else None
                    }
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found or invalid"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shared song: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting shared song: {str(e)}"
        )


@router.get("/{song_id}/shares")
async def get_song_shares(
    song_id: str,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene todos los enlaces de compartición de una canción.
    
    Args:
        song_id: ID único de la canción (UUID)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de enlaces de compartición activos
    
    Example:
        ```
        GET /suno/songs/123e4567-e89b-12d3-a456-426614174000/shares
        ```
    """
    try:
        validate_song_id(song_id)
        song = ensure_song_exists(song_service.get_song(song_id), song_id)
        
        shares = song.get("metadata", {}).get("shares", [])
        
        # Filtrar solo enlaces activos y no expirados
        active_shares = []
        for share in shares:
            if not share.get("is_active", True):
                continue
            
            expires_at_str = share.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    continue
            
            max_uses = share.get("max_uses")
            uses_count = share.get("uses_count", 0)
            if max_uses and uses_count >= max_uses:
                continue
            
            active_shares.append({
                **share,
                "share_link": f"/suno/songs/shared/{share.get('token')}",
                "uses_remaining": (max_uses - uses_count) if max_uses else None
            })
        
        return {
            "song_id": song_id,
            "total_shares": len(shares),
            "active_shares": len(active_shares),
            "shares": active_shares
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting song shares: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting song shares: {str(e)}"
        )


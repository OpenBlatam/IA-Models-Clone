"""
Rutas de Compartir Manuales
=============================

Endpoints para compartir manuales con tokens únicos.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.session import get_async_session
from ...services.share_service import ShareService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["share"])


# Modelos Pydantic
class ShareRequest(BaseModel):
    """Request para crear share."""
    expires_in_days: Optional[int] = None


class ShareResponse(BaseModel):
    """Response de share."""
    share_token: str
    share_url: str
    expires_at: Optional[str] = None
    manual_id: int


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.post("/manuals/{manual_id}/share", response_model=ShareResponse)
async def create_share(
    manual_id: int = Path(..., description="ID del manual"),
    request: ShareRequest = ...,
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Crear enlace para compartir manual.
    
    - **manual_id**: ID del manual
    - **expires_in_days**: Días hasta expiración (opcional)
    - **user_id**: ID del usuario que comparte (opcional)
    """
    try:
        service = ShareService(db)
        share = await service.create_share(
            manual_id=manual_id,
            shared_by=user_id,
            expires_in_days=request.expires_in_days
        )
        
        # Construir URL (ajustar según tu dominio)
        share_url = f"/api/v1/share/{share.share_token}"
        
        return ShareResponse(
            share_token=share.share_token,
            share_url=share_url,
            expires_at=share.expires_at.isoformat() if share.expires_at else None,
            manual_id=manual_id
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creando share: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creando share: {str(e)}")


@router.get("/share/{share_token}")
async def get_shared_manual(
    share_token: str = Path(..., description="Token de compartir"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener manual compartido por token.
    
    - **share_token**: Token de compartir
    """
    try:
        service = ShareService(db)
        manual = await service.get_manual_by_token(share_token)
        
        if not manual:
            raise HTTPException(status_code=404, detail="Enlace no encontrado o expirado")
        
        from ...api.routes.history import ManualDetailResponse
        
        return ManualDetailResponse.from_orm(manual)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo manual compartido: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo manual: {str(e)}")


@router.delete("/share/{share_token}")
async def revoke_share(
    share_token: str = Path(..., description="Token de compartir"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Revocar enlace de compartir.
    
    - **share_token**: Token de compartir
    """
    try:
        service = ShareService(db)
        revoked = await service.revoke_share(share_token)
        
        if not revoked:
            raise HTTPException(status_code=404, detail="Enlace no encontrado")
        
        return {
            "success": True,
            "message": "Enlace revocado exitosamente"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revocando share: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error revocando share: {str(e)}")


@router.get("/manuals/{manual_id}/share/stats")
async def get_share_stats(
    manual_id: int = Path(..., description="ID del manual"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener estadísticas de compartir.
    
    - **manual_id**: ID del manual
    """
    try:
        service = ShareService(db)
        stats = await service.get_share_stats(manual_id)
        
        return {
            "success": True,
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo stats: {str(e)}")





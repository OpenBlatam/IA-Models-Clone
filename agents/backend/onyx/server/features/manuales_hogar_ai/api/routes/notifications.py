"""
Rutas de Notificaciones
========================

Endpoints para gestionar notificaciones.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.session import get_async_session
from ...services.notification.notification_service import NotificationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["notifications"])


# Modelos Pydantic
class NotificationResponse(BaseModel):
    """Response de notificación."""
    id: int
    user_id: str
    manual_id: Optional[int]
    type: str
    title: str
    message: str
    is_read: bool
    read_at: Optional[str]
    created_at: str
    
    class Config:
        from_attributes = True


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    user_id: str = Query(..., description="ID del usuario"),
    unread_only: bool = Query(False, description="Solo no leídas"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener notificaciones de usuario.
    
    - **user_id**: ID del usuario
    - **unread_only**: Solo no leídas
    - **limit**: Límite de resultados
    - **offset**: Offset para paginación
    """
    try:
        service = NotificationService(db)
        notifications = await service.get_user_notifications(
            user_id=user_id,
            unread_only=unread_only,
            limit=limit,
            offset=offset
        )
        
        return [
            NotificationResponse(
                id=n.id,
                user_id=n.user_id,
                manual_id=n.manual_id,
                type=n.type,
                title=n.title,
                message=n.message,
                is_read=n.is_read,
                read_at=n.read_at.isoformat() if n.read_at else None,
                created_at=n.created_at.isoformat()
            )
            for n in notifications
        ]
    
    except Exception as e:
        logger.error(f"Error obteniendo notificaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo notificaciones: {str(e)}")


@router.get("/notifications/unread-count")
async def get_unread_count(
    user_id: str = Query(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener contador de notificaciones no leídas.
    
    - **user_id**: ID del usuario
    """
    try:
        service = NotificationService(db)
        count = await service.get_unread_count(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "unread_count": count
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo contador: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo contador: {str(e)}")


@router.post("/notifications/{notification_id}/read")
async def mark_as_read(
    notification_id: int = Path(..., description="ID de la notificación"),
    user_id: str = Query(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Marcar notificación como leída.
    
    - **notification_id**: ID de la notificación
    - **user_id**: ID del usuario
    """
    try:
        service = NotificationService(db)
        marked = await service.mark_as_read(notification_id, user_id)
        
        if not marked:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        
        return {
            "success": True,
            "message": "Notificación marcada como leída"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marcando notificación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error marcando notificación: {str(e)}")


@router.post("/notifications/mark-all-read")
async def mark_all_as_read(
    user_id: str = Query(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Marcar todas las notificaciones como leídas.
    
    - **user_id**: ID del usuario
    """
    try:
        service = NotificationService(db)
        count = await service.mark_all_as_read(user_id)
        
        return {
            "success": True,
            "message": f"{count} notificaciones marcadas como leídas",
            "count": count
        }
    
    except Exception as e:
        logger.error(f"Error marcando notificaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error marcando notificaciones: {str(e)}")


@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: int = Path(..., description="ID de la notificación"),
    user_id: str = Query(..., description="ID del usuario"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Eliminar notificación.
    
    - **notification_id**: ID de la notificación
    - **user_id**: ID del usuario
    """
    try:
        service = NotificationService(db)
        deleted = await service.delete_notification(notification_id, user_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        
        return {
            "success": True,
            "message": "Notificación eliminada exitosamente"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando notificación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error eliminando notificación: {str(e)}")





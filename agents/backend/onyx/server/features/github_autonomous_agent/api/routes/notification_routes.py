"""
Notification Routes - Rutas para gestionar notificaciones.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services import NotificationService, NotificationLevel, NotificationChannel
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class CreateNotificationRequest(BaseModel):
    """Request para crear notificación."""
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=2000)
    level: str = Field(default="info", description="Nivel: info, warning, error, critical, success")
    channels: Optional[List[str]] = Field(default=None, description="Canales: log, websocket, email, webhook")


@router.post("/")
@handle_api_errors
async def create_notification(request: CreateNotificationRequest):
    """
    Crear y enviar notificación.
    
    Args:
        request: Datos de la notificación
        
    Returns:
        Notificación creada
    """
    try:
        notification_service: NotificationService = get_service("notification_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Notification service no disponible")
    
    # Validar level
    try:
        level = NotificationLevel(request.level)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Nivel inválido: {request.level}. Válidos: {[l.value for l in NotificationLevel]}"
        )
    
    # Validar channels
    channels = None
    if request.channels:
        try:
            channels = [NotificationChannel(c) for c in request.channels]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Canal inválido. Válidos: {[c.value for c in NotificationChannel]}"
            )
    
    notification = await notification_service.send(
        title=request.title,
        message=request.message,
        level=level,
        channels=channels
    )
    
    return notification.to_dict()


@router.get("/")
@handle_api_errors
async def get_notifications(
    level: Optional[str] = Query(None, description="Filtrar por nivel"),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Obtener notificaciones.
    
    Args:
        level: Nivel a filtrar
        limit: Número máximo de notificaciones
        
    Returns:
        Lista de notificaciones
    """
    try:
        notification_service: NotificationService = get_service("notification_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Notification service no disponible")
    
    level_enum = None
    if level:
        try:
            level_enum = NotificationLevel(level)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Nivel inválido: {level}"
            )
    
    notifications = notification_service.get_notifications(
        level=level_enum,
        limit=limit
    )
    
    return {
        "total": len(notifications),
        "notifications": [n.to_dict() for n in notifications]
    }


@router.get("/stats")
@handle_api_errors
async def get_notification_stats():
    """
    Obtener estadísticas de notificaciones.
    
    Returns:
        Estadísticas de notificaciones
    """
    try:
        notification_service: NotificationService = get_service("notification_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Notification service no disponible")
    
    return notification_service.get_stats()




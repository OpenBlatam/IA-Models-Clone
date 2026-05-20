"""
Notification Routes - Rutas para notificaciones
================================================

Endpoints para consultar y gestionar notificaciones.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Query

from ..utils import get_agent, handle_route_errors, AgentDep, create_success_response
from ..serializers import serialize_notifications

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


@router.get("")
@handle_route_errors("getting notifications")
async def get_notifications(
    limit: int = Query(50, ge=1, le=1000, description="Número máximo de notificaciones"),
    unread_only: bool = Query(False, description="Solo notificaciones no leídas"),
    agent = AgentDep
):
    """
    Obtener notificaciones.
    
    Args:
        limit: Número máximo de notificaciones a retornar.
        unread_only: Si True, solo retorna notificaciones no leídas.
        agent: Instancia del agente (inyectada).
    
    Returns:
        Lista de notificaciones.
    
    Raises:
        HTTPException: Si hay error al obtener las notificaciones.
    """
    if not agent.notifications:
        return {"notifications": []}
    
    notifications = agent.notifications.get_notifications(
        unread_only=unread_only,
        limit=limit
    )
    
    return {
        "notifications": serialize_notifications(notifications)
    }


@router.post("/{notification_id}/read")
@handle_route_errors("marking notification as read")
async def mark_notification_read(
    notification_id: str,
    agent = AgentDep
):
    """
    Marcar notificación como leída.
    
    Args:
        notification_id: ID de la notificación.
        agent: Instancia del agente (inyectada).
    
    Returns:
        Mensaje de confirmación.
    
    Raises:
        HTTPException: Si hay error al marcar la notificación.
    """
    if not agent.notifications:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Notifications not available")
    
    try:
        agent.notifications.mark_as_read(notification_id)
        return create_success_response(
            "success",
            "Notification marked as read"
        )
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Notification not found")


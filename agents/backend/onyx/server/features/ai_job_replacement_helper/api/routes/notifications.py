"""
Notifications endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.notifications import NotificationsService

router = APIRouter()
notifications_service = NotificationsService()


@router.get("/{user_id}")
async def get_notifications(
    user_id: str,
    unread_only: bool = False,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Obtener notificaciones del usuario"""
    try:
        notifications = notifications_service.get_user_notifications(
            user_id, unread_only, limit
        )
        return {
            "notifications": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "title": n.title,
                    "message": n.message,
                    "priority": n.priority.value,
                    "read": n.read,
                    "created_at": n.created_at.isoformat(),
                    "action_url": n.action_url,
                }
                for n in notifications
            ],
            "total": len(notifications),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unread-count/{user_id}")
async def get_unread_count(user_id: str) -> Dict[str, Any]:
    """Obtener cantidad de notificaciones no leídas"""
    try:
        count = notifications_service.get_unread_count(user_id)
        return {"unread_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark-read/{user_id}/{notification_id}")
async def mark_as_read(user_id: str, notification_id: str) -> Dict[str, Any]:
    """Marcar notificación como leída"""
    try:
        success = notifications_service.mark_as_read(user_id, notification_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark-all-read/{user_id}")
async def mark_all_as_read(user_id: str) -> Dict[str, Any]:
    """Marcar todas las notificaciones como leídas"""
    try:
        count = notifications_service.mark_all_as_read(user_id)
        return {"marked_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





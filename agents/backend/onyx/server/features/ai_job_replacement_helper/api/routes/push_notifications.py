"""
Push Notifications endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.push_notifications import PushNotificationsService, PushPlatform, PushPriority

router = APIRouter()
push_service = PushNotificationsService()


@router.post("/register-device/{user_id}")
async def register_device(
    user_id: str,
    device_id: str,
    platform: str,
    token: str
) -> Dict[str, Any]:
    """Registrar dispositivo para push"""
    try:
        platform_enum = PushPlatform(platform)
        device = push_service.register_device(user_id, device_id, platform_enum, token)
        return {
            "device_id": device.device_id,
            "platform": device.platform.value,
            "enabled": device.enabled,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send/{user_id}")
async def send_push(
    user_id: str,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
    priority: str = "normal"
) -> Dict[str, Any]:
    """Enviar notificación push"""
    try:
        priority_enum = PushPriority(priority)
        notification = push_service.send_push(user_id, title, body, data, priority_enum)
        return {
            "id": notification.id,
            "title": notification.title,
            "body": notification.body,
            "sent_at": notification.sent_at.isoformat() if notification.sent_at else None,
            "delivered": notification.delivered,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_notifications(
    user_id: str,
    limit: int = 50
) -> Dict[str, Any]:
    """Obtener notificaciones del usuario"""
    try:
        notifications = push_service.get_user_notifications(user_id, limit)
        return {
            "user_id": user_id,
            "notifications": notifications,
            "total": len(notifications),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/click/{notification_id}")
async def mark_as_clicked(notification_id: str) -> Dict[str, Any]:
    """Marcar notificación como clickeada"""
    try:
        success = push_service.mark_as_clicked(notification_id)
        return {"success": success, "notification_id": notification_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





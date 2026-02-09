"""
Notifications API Endpoints
===========================

Endpoints para notificaciones.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
import logging

try:
    from ..core.notification_system import (
        get_notification_system,
        NotificationType,
        NotificationChannel
    )
except ImportError:
    def get_notification_system():
        return None
    NotificationType = None
    NotificationChannel = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


@router.get("/")
async def get_notifications(
    notification_type: Optional[str] = None,
    unread_only: bool = False,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Obtener notificaciones."""
    try:
        system = get_notification_system()
        
        ntype = None
        if notification_type:
            try:
                ntype = NotificationType(notification_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid notification type: {notification_type}"
                )
        
        notifications = system.get_notifications(
            notification_type=ntype,
            unread_only=unread_only,
            limit=limit
        )
        
        return {
            "notifications": [
                {
                    "notification_id": n.notification_id,
                    "title": n.title,
                    "message": n.message,
                    "type": n.notification_type.value,
                    "channels": [c.value for c in n.channels],
                    "read": n.read,
                    "timestamp": n.timestamp,
                    "metadata": n.metadata
                }
                for n in notifications
            ],
            "count": len(notifications)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting notifications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def send_notification(
    title: str,
    message: str,
    notification_type: str = "info",
    channels: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Enviar notificación."""
    try:
        system = get_notification_system()
        
        try:
            ntype = NotificationType(notification_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid notification type: {notification_type}"
            )
        
        notification_channels = []
        if channels:
            for channel_str in channels:
                try:
                    notification_channels.append(NotificationChannel(channel_str))
                except ValueError:
                    logger.warning(f"Invalid channel: {channel_str}")
        
        notification = system.send_notification(
            title=title,
            message=message,
            notification_type=ntype,
            channels=notification_channels if notification_channels else None,
            metadata=metadata
        )
        
        return {
            "notification_id": notification.notification_id,
            "title": notification.title,
            "message": notification.message,
            "type": notification.notification_type.value,
            "timestamp": notification.timestamp
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending notification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{notification_id}/read")
async def mark_notification_read(notification_id: str) -> Dict[str, Any]:
    """Marcar notificación como leída."""
    try:
        system = get_notification_system()
        if system.mark_as_read(notification_id):
            return {"message": "Notification marked as read"}
        raise HTTPException(status_code=404, detail="Notification not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/read-all")
async def mark_all_read() -> Dict[str, Any]:
    """Marcar todas las notificaciones como leídas."""
    try:
        system = get_notification_system()
        count = system.mark_all_as_read()
        return {"message": f"Marked {count} notifications as read"}
    except Exception as e:
        logger.error(f"Error marking all as read: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_notification_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de notificaciones."""
    try:
        system = get_notification_system()
        stats = system.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))







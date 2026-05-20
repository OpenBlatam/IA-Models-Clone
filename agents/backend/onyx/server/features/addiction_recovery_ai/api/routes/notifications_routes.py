"""
Notifications routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from services.notification_service import NotificationService
except ImportError:
    from ...services.notification_service import NotificationService

router = APIRouter()

notifications = NotificationService()


@router.get("/notifications/{user_id}")
async def get_notifications(user_id: str):
    """Obtiene notificaciones pendientes del usuario"""
    try:
        pending = notifications.get_pending_notifications(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "notifications": pending,
            "unread_count": len([n for n in pending if not n.get("read", False)]),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo notificaciones: {str(e)}")


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Marca una notificación como leída"""
    try:
        success = notifications.mark_notification_read(notification_id)
        return JSONResponse(content={
            "notification_id": notification_id,
            "marked_read": success,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marcando notificación: {str(e)}")


@router.get("/reminders/{user_id}")
async def get_reminders(user_id: str):
    """Obtiene recordatorios diarios del usuario"""
    try:
        reminders = notifications.get_daily_reminders(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "reminders": reminders,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recordatorios: {str(e)}")




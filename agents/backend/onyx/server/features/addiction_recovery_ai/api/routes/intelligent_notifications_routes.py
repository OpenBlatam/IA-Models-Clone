"""
Intelligent notifications routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.intelligent_notifications_service import IntelligentNotificationsService
except ImportError:
    from ...services.intelligent_notifications_service import IntelligentNotificationsService

router = APIRouter()

intelligent_notifications = IntelligentNotificationsService()


@router.post("/notifications/intelligent/send")
async def send_intelligent_notification(
    user_id: str = Body(...),
    notification_data: Dict = Body(...)
):
    """Envía notificación inteligente"""
    try:
        notification = intelligent_notifications.send_intelligent_notification(
            user_id, notification_data
        )
        return JSONResponse(content=notification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enviando notificación: {str(e)}")




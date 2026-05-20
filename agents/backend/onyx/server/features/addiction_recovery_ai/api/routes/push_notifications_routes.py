"""
Push notifications routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse

try:
    from services.push_notification_service import PushNotificationService
except ImportError:
    from ...services.push_notification_service import PushNotificationService

router = APIRouter()

push_notifications = PushNotificationService()


@router.post("/push/register-device")
async def register_push_device(
    user_id: str = Body(...),
    device_token: str = Body(...),
    platform: str = Body(...)
):
    """Registra dispositivo para notificaciones push"""
    try:
        device = push_notifications.register_device(user_id, device_token, platform)
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")


@router.post("/push/send")
async def send_push_notification(
    user_id: str = Body(...),
    title: str = Body(...),
    body: str = Body(...),
    priority: str = Body("normal")
):
    """Envía una notificación push"""
    try:
        notification = push_notifications.send_notification(user_id, title, body, priority)
        return JSONResponse(content=notification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enviando notificación: {str(e)}")




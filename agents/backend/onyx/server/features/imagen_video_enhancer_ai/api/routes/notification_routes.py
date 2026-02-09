"""
Notification Routes
===================

API routes for notifications.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..dependencies import get_notification_manager, verify_api_key
from ...core.notification_system import Notification, NotificationChannel, NotificationPriority

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.post("/send")
async def send_notification(
    title: str,
    message: str,
    channel: str,
    priority: str = "normal",
    metadata: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Send a notification."""
    notification_manager = get_notification_manager()
    
    if api_key:
        from ..dependencies import get_auth_manager
        auth_manager = get_auth_manager()
        if not auth_manager.check_permission(api_key, "notifications"):
            raise HTTPException(status_code=403, detail="Notification permission required")
    
    try:
        notification = Notification(
            title=title,
            message=message,
            channel=NotificationChannel(channel),
            priority=NotificationPriority(priority),
            metadata=metadata or {}
        )
        
        result = await notification_manager.send(notification)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error sending notification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_notification_stats(api_key: Optional[str] = Depends(verify_api_key)):
    """Get notification statistics."""
    notification_manager = get_notification_manager()
    
    if api_key:
        from ..dependencies import get_auth_manager
        auth_manager = get_auth_manager()
        if not auth_manager.check_permission(api_key, "admin"):
            raise HTTPException(status_code=403, detail="Admin permission required")
    
    try:
        stats = notification_manager.get_stats()
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))





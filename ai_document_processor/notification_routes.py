"""
Notification Routes
Real, working notification endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from notification_system import notification_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])

@router.post("/send-notification")
async def send_notification(
    notification_type: str = Form(...),
    message: str = Form(...),
    recipients: Optional[List[str]] = Form(None)
):
    """Send notification to subscribers"""
    try:
        result = await notification_system.send_notification(
            notification_type, message, recipients
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/subscribe")
async def subscribe(
    subscriber_id: str = Form(...),
    notification_types: List[str] = Form(...),
    email: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None)
):
    """Subscribe to notifications"""
    try:
        result = await notification_system.subscribe(
            subscriber_id, notification_types, email, webhook_url
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error subscribing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unsubscribe")
async def unsubscribe(
    subscriber_id: str = Form(...)
):
    """Unsubscribe from notifications"""
    try:
        result = await notification_system.unsubscribe(subscriber_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error unsubscribing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-processing-notification")
async def send_processing_notification(
    processing_result: dict,
    recipients: Optional[List[str]] = Form(None)
):
    """Send processing completion notification"""
    try:
        result = await notification_system.send_processing_notification(
            processing_result, recipients
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error sending processing notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-error-notification")
async def send_error_notification(
    error_message: str = Form(...),
    recipients: Optional[List[str]] = Form(None)
):
    """Send error notification"""
    try:
        result = await notification_system.send_error_notification(
            error_message, recipients
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error sending error notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-security-notification")
async def send_security_notification(
    security_event: str = Form(...),
    client_ip: str = Form(...),
    recipients: Optional[List[str]] = Form(None)
):
    """Send security notification"""
    try:
        result = await notification_system.send_security_notification(
            security_event, client_ip, recipients
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error sending security notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-performance-notification")
async def send_performance_notification(
    performance_data: dict,
    recipients: Optional[List[str]] = Form(None)
):
    """Send performance notification"""
    try:
        result = await notification_system.send_performance_notification(
            performance_data, recipients
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error sending performance notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notifications")
async def get_notifications(
    limit: int = 50
):
    """Get recent notifications"""
    try:
        notifications = notification_system.get_notifications(limit)
        return JSONResponse(content={
            "notifications": notifications,
            "total_count": len(notification_system.notifications)
        })
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subscribers")
async def get_subscribers():
    """Get all subscribers"""
    try:
        subscribers = notification_system.get_subscribers()
        return JSONResponse(content={
            "subscribers": subscribers,
            "total_count": len(subscribers),
            "active_count": len([s for s in subscribers.values() if s["active"]])
        })
    except Exception as e:
        logger.error(f"Error getting subscribers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notification-stats")
async def get_notification_stats():
    """Get notification statistics"""
    try:
        stats = notification_system.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notification-config")
async def get_notification_config():
    """Get notification configuration"""
    try:
        config = notification_system.get_config()
        return JSONResponse(content=config)
    except Exception as e:
        logger.error(f"Error getting notification config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-notifications")
async def health_check_notifications():
    """Notification system health check"""
    try:
        stats = notification_system.get_stats()
        config = notification_system.get_config()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Notification System",
            "version": "1.0.0",
            "features": {
                "email_notifications": config["features"]["email_notifications"],
                "webhook_notifications": config["features"]["webhook_notifications"],
                "processing_notifications": config["features"]["processing_notifications"],
                "error_notifications": config["features"]["error_notifications"],
                "security_notifications": config["features"]["security_notifications"],
                "performance_notifications": config["features"]["performance_notifications"]
            },
            "notification_stats": stats["stats"],
            "subscribers": {
                "total": stats["total_subscribers"],
                "active": stats["active_subscribers"]
            },
            "email_config": config["email_config"]
        })
    except Exception as e:
        logger.error(f"Error in notification health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))














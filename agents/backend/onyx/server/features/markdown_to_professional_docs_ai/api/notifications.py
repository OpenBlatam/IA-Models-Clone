"""Notifications endpoints"""
from fastapi import APIRouter, HTTPException, Form
from typing import Optional
from utils.notifications import get_notification_manager

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.post("/send")
async def send_notification(
    recipient: str = Form(...),
    notification_type: str = Form(...),
    message: str = Form(...),
    metadata: Optional[str] = Form(None)  # JSON string
):
    """Send notification"""
    import json
    
    notification_manager = get_notification_manager()
    
    metadata_dict = None
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            pass
    
    notification_manager.send_notification(
        recipient,
        notification_type,
        message,
        metadata_dict
    )
    
    return {
        "status": "success",
        "recipient": recipient,
        "notification_type": notification_type
    }


@router.get("/{recipient}")
async def get_notifications(recipient: str):
    """Get notifications for recipient"""
    notification_manager = get_notification_manager()
    notifications = notification_manager.get_notifications(recipient)
    
    return {
        "recipient": recipient,
        "notifications": notifications,
        "total": len(notifications)
    }


@router.get("/{recipient}/unread")
async def get_unread_notifications(recipient: str):
    """Get unread notifications"""
    notification_manager = get_notification_manager()
    notifications = notification_manager.get_unread_notifications(recipient)
    
    return {
        "recipient": recipient,
        "unread_count": len(notifications),
        "notifications": notifications
    }


@router.post("/{recipient}/mark_read")
async def mark_notifications_read(
    recipient: str,
    notification_ids: Optional[str] = Form(None)  # Comma-separated IDs
):
    """Mark notifications as read"""
    import json
    
    notification_manager = get_notification_manager()
    
    ids = None
    if notification_ids:
        ids = [id.strip() for id in notification_ids.split(",")]
    
    notification_manager.mark_as_read(recipient, ids)
    
    return {
        "status": "success",
        "recipient": recipient
    }


"""
Notifications API endpoints.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends
from pydantic import BaseModel, Field
from typing import Dict, Any

from .base_router import BaseRouter
from ..core.notifications import NotificationManager
from ..utils.data_helpers import count_matching

# Create base router instance
base = BaseRouter(
    prefix="/api/notifications",
    tags=["Notifications"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router

notification_manager = NotificationManager()


class GetNotificationsRequest(BaseModel):
    """Request to get notifications."""
    user_id: str = Field(..., description="User identifier")
    unread_only: bool = Field(False, description="Only return unread notifications")
    limit: int = Field(50, description="Maximum number of notifications")


class MarkReadRequest(BaseModel):
    """Request to mark notification as read."""
    user_id: str = Field(..., description="User identifier")
    notification_index: int = Field(..., description="Index of notification")


@router.get("/{user_id}")
@base.timed_endpoint("get_notifications")
async def get_notifications(
    user_id: str,
    unread_only: bool = False,
    limit: int = 50,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """Get notifications for a user."""
    base.log_request("get_notifications", user_id=user_id, unread_only=unread_only)
    
    notifications = notification_manager.get_notifications(
        user_id=user_id,
        unread_only=unread_only,
        limit=limit
    )
    return base.success({
        "notifications": [n.to_dict() for n in notifications],
        "count": len(notifications),
        "unread_count": count_matching(notifications, lambda n: not n.read)
    })


@router.post("/mark-read")
@base.timed_endpoint("mark_notification_read")
async def mark_notification_read(
    request: MarkReadRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """Mark a notification as read."""
    base.log_request("mark_notification_read", user_id=request.user_id)
    
    notification_manager.mark_as_read(
        user_id=request.user_id,
        notification_index=request.notification_index
    )
    return base.success(None, message="Notification marked as read")


@router.delete("/{user_id}")
@base.timed_endpoint("clear_notifications")
async def clear_notifications(
    user_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """Clear all notifications for a user."""
    base.log_request("clear_notifications", user_id=user_id)
    
    notification_manager.clear_notifications(user_id)
    return base.success(None, message="Notifications cleared")







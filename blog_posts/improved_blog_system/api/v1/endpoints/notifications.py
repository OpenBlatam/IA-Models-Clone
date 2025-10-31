"""
Advanced Notification API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_notification_service import AdvancedNotificationService, NotificationPriority
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class SendNotificationRequest(BaseModel):
    """Request model for sending notifications."""
    recipient: str = Field(..., description="Recipient identifier")
    template_name: str = Field(..., description="Notification template name")
    data: Dict[str, Any] = Field(default_factory=dict, description="Template data")
    channels: Optional[List[str]] = Field(default=None, description="Notification channels")
    priority: str = Field(default="normal", description="Notification priority")


class ScheduleNotificationRequest(BaseModel):
    """Request model for scheduling notifications."""
    recipient: str = Field(..., description="Recipient identifier")
    template_name: str = Field(..., description="Notification template name")
    data: Dict[str, Any] = Field(default_factory=dict, description="Template data")
    scheduled_at: datetime = Field(..., description="Scheduled delivery time")
    channels: Optional[List[str]] = Field(default=None, description="Notification channels")
    priority: str = Field(default="normal", description="Notification priority")


class BroadcastNotificationRequest(BaseModel):
    """Request model for broadcasting notifications."""
    template_name: str = Field(..., description="Notification template name")
    data: Dict[str, Any] = Field(default_factory=dict, description="Template data")
    channels: Optional[List[str]] = Field(default=None, description="Notification channels")
    priority: str = Field(default="normal", description="Notification priority")


class ConfigureChannelRequest(BaseModel):
    """Request model for configuring notification channels."""
    channel_name: str = Field(..., description="Channel name")
    config: Dict[str, Any] = Field(..., description="Channel configuration")
    enabled: bool = Field(default=True, description="Enable/disable channel")


class CreateTemplateRequest(BaseModel):
    """Request model for creating custom templates."""
    template_name: str = Field(..., description="Template name")
    subject: str = Field(..., description="Template subject")
    body: str = Field(..., description="Template body")
    channels: List[str] = Field(..., description="Notification channels")


async def get_notification_service(session: DatabaseSessionDep) -> AdvancedNotificationService:
    """Get notification service instance."""
    return AdvancedNotificationService(session)


@router.post("/send", response_model=Dict[str, Any])
async def send_notification(
    request: SendNotificationRequest = Depends(),
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Send notification using template."""
    try:
        priority = NotificationPriority[request.priority.upper()]
        
        result = await notification_service.send_notification(
            recipient=request.recipient,
            template_name=request.template_name,
            data=request.data,
            channels=request.channels,
            priority=priority
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Notification sent successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send notification"
        )


@router.post("/schedule", response_model=Dict[str, Any])
async def schedule_notification(
    request: ScheduleNotificationRequest = Depends(),
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Schedule notification for later delivery."""
    try:
        priority = NotificationPriority[request.priority.upper()]
        
        result = await notification_service.schedule_notification(
            recipient=request.recipient,
            template_name=request.template_name,
            data=request.data,
            scheduled_at=request.scheduled_at,
            channels=request.channels,
            priority=priority
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Notification scheduled successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule notification"
        )


@router.post("/broadcast", response_model=Dict[str, Any])
async def broadcast_notification(
    request: BroadcastNotificationRequest = Depends(),
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Broadcast notification to all connected users."""
    try:
        priority = NotificationPriority[request.priority.upper()]
        
        result = await notification_service.broadcast_notification(
            template_name=request.template_name,
            data=request.data,
            channels=request.channels,
            priority=priority
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Notification broadcast successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast notification"
        )


@router.post("/process-scheduled", response_model=Dict[str, Any])
async def process_scheduled_notifications(
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Process scheduled notifications that are due."""
    try:
        result = await notification_service.process_scheduled_notifications()
        
        return {
            "success": True,
            "data": result,
            "message": "Scheduled notifications processed successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process scheduled notifications"
        )


@router.get("/history", response_model=Dict[str, Any])
async def get_notification_history(
    recipient: Optional[str] = Query(None, description="Filter by recipient"),
    template_name: Optional[str] = Query(None, description="Filter by template"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of notifications to return"),
    offset: int = Query(default=0, ge=0, description="Number of notifications to skip"),
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get notification history."""
    try:
        result = await notification_service.get_notification_history(
            recipient=recipient,
            template_name=template_name,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Notification history retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get notification history"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_notification_stats(
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get notification statistics."""
    try:
        stats = await notification_service.get_notification_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Notification statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get notification statistics"
        )


@router.post("/channels/configure", response_model=Dict[str, Any])
async def configure_channel(
    request: ConfigureChannelRequest = Depends(),
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Configure notification channel."""
    try:
        result = await notification_service.configure_channel(
            channel_name=request.channel_name,
            config=request.config,
            enabled=request.enabled
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Channel configured successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure channel"
        )


@router.post("/channels/{channel_name}/test", response_model=Dict[str, Any])
async def test_channel(
    channel_name: str,
    test_recipient: str = Query(..., description="Test recipient"),
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Test notification channel."""
    try:
        result = await notification_service.test_channel(
            channel_name=channel_name,
            test_recipient=test_recipient
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Channel test completed"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test channel"
        )


@router.get("/templates", response_model=Dict[str, Any])
async def get_available_templates(
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get available notification templates."""
    try:
        result = await notification_service.get_available_templates()
        
        return {
            "success": True,
            "data": result,
            "message": "Templates retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get templates"
        )


@router.post("/templates/create", response_model=Dict[str, Any])
async def create_custom_template(
    request: CreateTemplateRequest = Depends(),
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Create custom notification template."""
    try:
        result = await notification_service.create_custom_template(
            template_name=request.template_name,
            subject=request.subject,
            body=request.body,
            channels=request.channels
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Template created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create template"
        )


@router.get("/channels", response_model=Dict[str, Any])
async def get_available_channels(
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get available notification channels."""
    try:
        channels_info = {}
        for name, channel in notification_service.channels.items():
            channels_info[name] = {
                "name": channel.name,
                "type": channel.type.value,
                "enabled": channel.enabled,
                "rate_limit": channel.rate_limit,
                "priority_threshold": channel.priority_threshold.value
            }
        
        return {
            "success": True,
            "data": {
                "channels": channels_info,
                "total": len(channels_info)
            },
            "message": "Channels retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get channels"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_notification_health(
    notification_service: AdvancedNotificationService = Depends(get_notification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get notification system health status."""
    try:
        # Get notification stats
        stats = await notification_service.get_notification_stats()
        
        # Calculate health metrics
        total_notifications = stats.get("total_notifications", 0)
        status_stats = stats.get("status_stats", {})
        failed_count = status_stats.get("failed", 0)
        
        # Calculate health score
        health_score = 100
        if total_notifications > 0:
            failure_rate = (failed_count / total_notifications) * 100
            if failure_rate > 10:
                health_score -= 30
            elif failure_rate > 5:
                health_score -= 15
        
        # Check channel status
        enabled_channels = sum(1 for ch in notification_service.channels.values() if ch.enabled)
        total_channels = len(notification_service.channels)
        
        if enabled_channels < total_channels * 0.5:
            health_score -= 20
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_notifications": total_notifications,
                "failed_notifications": failed_count,
                "enabled_channels": enabled_channels,
                "total_channels": total_channels,
                "connected_users": len(notification_service.websocket_connections),
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Notification health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get notification health status"
        )

























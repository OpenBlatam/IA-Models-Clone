"""
Notification API routes for Facebook Posts API
Real-time notifications, alerts, and communication system
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
import structlog

from ..core.config import get_settings
from ..api.schemas import ErrorResponse
from ..api.dependencies import (
    get_current_user, check_rate_limit, get_request_id
)
from ..services.notification_service import (
    get_notification_service, NotificationType, NotificationPriority,
    NotificationStatus, NotificationTemplate, NotificationRule
)
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])


# Notification Sending Routes

@router.post(
    "/send",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Notification sent successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Notification sending error"}
    },
    summary="Send notification",
    description="Send a notification via email, webhook, Slack, or other channels"
)
@timed("notification_send")
async def send_notification(
    notification_type: str = Query(..., description="Notification type"),
    priority: str = Query(..., description="Notification priority"),
    title: str = Query(..., description="Notification title"),
    message: str = Query(..., description="Notification message"),
    recipient: str = Query(..., description="Notification recipient"),
    metadata: Optional[Dict[str, Any]] = Query(None, description="Additional metadata"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Send a notification"""
    
    # Validate notification type
    valid_notification_types = [nt.value for nt in NotificationType]
    if notification_type not in valid_notification_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid notification type. Valid types: {valid_notification_types}"
        )
    
    # Validate priority
    valid_priorities = [np.value for np in NotificationPriority]
    if priority not in valid_priorities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority. Valid priorities: {valid_priorities}"
        )
    
    # Validate required fields
    if not title or not message or not recipient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title, message, and recipient are required"
        )
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Send notification
        notification_id = await notification_service.send_notification(
            notification_type=NotificationType(notification_type),
            priority=NotificationPriority(priority),
            title=title,
            message=message,
            recipient=recipient,
            metadata=metadata
        )
        
        logger.info(
            "Notification sent",
            notification_id=notification_id,
            notification_type=notification_type,
            priority=priority,
            recipient=recipient,
            request_id=request_id
        )
        
        return {
            "success": True,
            "notification_id": notification_id,
            "status": "queued",
            "metadata": {
                "notification_type": notification_type,
                "priority": priority,
                "recipient": recipient,
                "title": title
            },
            "request_id": request_id,
            "sent_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Notification sending failed",
            notification_type=notification_type,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Notification sending failed: {str(e)}"
        )


@router.post(
    "/send/template",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Template notification sent successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Template notification error"}
    },
    summary="Send template notification",
    description="Send a notification using a predefined template"
)
@timed("template_notification_send")
async def send_template_notification(
    template_id: str = Query(..., description="Template ID"),
    recipient: str = Query(..., description="Notification recipient"),
    variables: Dict[str, Any] = Query(..., description="Template variables"),
    notification_type: Optional[str] = Query(None, description="Override notification type"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Send notification using template"""
    
    # Validate notification type if provided
    if notification_type:
        valid_notification_types = [nt.value for nt in NotificationType]
        if notification_type not in valid_notification_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid notification type. Valid types: {valid_notification_types}"
            )
    
    # Validate required fields
    if not template_id or not recipient or not variables:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Template ID, recipient, and variables are required"
        )
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Send template notification
        notification_id = await notification_service.send_template_notification(
            template_id=template_id,
            recipient=recipient,
            variables=variables,
            notification_type=NotificationType(notification_type) if notification_type else None
        )
        
        logger.info(
            "Template notification sent",
            notification_id=notification_id,
            template_id=template_id,
            recipient=recipient,
            request_id=request_id
        )
        
        return {
            "success": True,
            "notification_id": notification_id,
            "template_id": template_id,
            "status": "queued",
            "metadata": {
                "recipient": recipient,
                "variables": variables,
                "notification_type": notification_type
            },
            "request_id": request_id,
            "sent_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Template notification sending failed",
            template_id=template_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template notification sending failed: {str(e)}"
        )


@router.get(
    "/status/{notification_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Notification status retrieved successfully"},
        404: {"description": "Notification not found"},
        500: {"description": "Status retrieval error"}
    },
    summary="Get notification status",
    description="Get the status of a notification"
)
@timed("notification_status")
async def get_notification_status(
    notification_id: str = Path(..., description="Notification ID"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get notification status"""
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Get notification status
        status = await notification_service.get_notification_status(notification_id)
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        logger.info(
            "Notification status retrieved",
            notification_id=notification_id,
            status=status["status"],
            request_id=request_id
        )
        
        return {
            "success": True,
            "notification_id": notification_id,
            "status": status,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Notification status retrieval failed",
            notification_id=notification_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Notification status retrieval failed: {str(e)}"
        )


# Template Management Routes

@router.get(
    "/templates",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Templates retrieved successfully"},
        500: {"description": "Template retrieval error"}
    },
    summary="Get notification templates",
    description="Get all available notification templates"
)
@timed("notification_templates")
async def get_notification_templates(
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get all notification templates"""
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Get templates
        templates = []
        for template_id, template in notification_service.template_manager.templates.items():
            templates.append({
                "id": template.id,
                "name": template.name,
                "type": template.type.value,
                "subject_template": template.subject_template,
                "body_template": template.body_template,
                "variables": template.variables,
                "metadata": template.metadata
            })
        
        logger.info(
            "Notification templates retrieved",
            templates_count=len(templates),
            request_id=request_id
        )
        
        return {
            "success": True,
            "templates": templates,
            "total_count": len(templates),
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Notification templates retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Notification templates retrieval failed: {str(e)}"
        )


@router.get(
    "/templates/{template_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Template retrieved successfully"},
        404: {"description": "Template not found"},
        500: {"description": "Template retrieval error"}
    },
    summary="Get notification template",
    description="Get a specific notification template by ID"
)
@timed("notification_template")
async def get_notification_template(
    template_id: str = Path(..., description="Template ID"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get notification template by ID"""
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Get template
        template = await notification_service.template_manager.get_template(template_id)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        logger.info(
            "Notification template retrieved",
            template_id=template_id,
            request_id=request_id
        )
        
        return {
            "success": True,
            "template": {
                "id": template.id,
                "name": template.name,
                "type": template.type.value,
                "subject_template": template.subject_template,
                "body_template": template.body_template,
                "variables": template.variables,
                "metadata": template.metadata
            },
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Notification template retrieval failed",
            template_id=template_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Notification template retrieval failed: {str(e)}"
        )


@router.post(
    "/templates",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Template created successfully"},
        400: {"description": "Invalid template data"},
        500: {"description": "Template creation error"}
    },
    summary="Create notification template",
    description="Create a new notification template"
)
@timed("notification_template_create")
async def create_notification_template(
    template_data: Dict[str, Any],
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create notification template"""
    
    try:
        # Validate required fields
        required_fields = ["id", "name", "type", "subject_template", "body_template"]
        for field in required_fields:
            if field not in template_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )
        
        # Validate notification type
        valid_notification_types = [nt.value for nt in NotificationType]
        if template_data["type"] not in valid_notification_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid notification type. Valid types: {valid_notification_types}"
            )
        
        # Create template
        template = NotificationTemplate(
            id=template_data["id"],
            name=template_data["name"],
            type=NotificationType(template_data["type"]),
            subject_template=template_data["subject_template"],
            body_template=template_data["body_template"],
            variables=template_data.get("variables", []),
            metadata=template_data.get("metadata", {})
        )
        
        # Get notification service
        notification_service = get_notification_service()
        
        # Create template
        success = await notification_service.template_manager.create_template(template)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create template"
            )
        
        logger.info(
            "Notification template created",
            template_id=template.id,
            request_id=request_id
        )
        
        return {
            "success": True,
            "template_id": template.id,
            "template": {
                "id": template.id,
                "name": template.name,
                "type": template.type.value,
                "subject_template": template.subject_template,
                "body_template": template.body_template,
                "variables": template.variables,
                "metadata": template.metadata
            },
            "request_id": request_id,
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Notification template creation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Notification template creation failed: {str(e)}"
        )


# Rule Management Routes

@router.get(
    "/rules",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Rules retrieved successfully"},
        500: {"description": "Rule retrieval error"}
    },
    summary="Get notification rules",
    description="Get all notification rules"
)
@timed("notification_rules")
async def get_notification_rules(
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get all notification rules"""
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Get rules
        rules = []
        for rule_id, rule in notification_service.rule_engine.rules.items():
            rules.append({
                "id": rule.id,
                "name": rule.name,
                "condition": rule.condition,
                "notification_template_id": rule.notification_template_id,
                "recipients": rule.recipients,
                "enabled": rule.enabled,
                "cooldown_minutes": rule.cooldown_minutes,
                "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
                "metadata": rule.metadata
            })
        
        logger.info(
            "Notification rules retrieved",
            rules_count=len(rules),
            request_id=request_id
        )
        
        return {
            "success": True,
            "rules": rules,
            "total_count": len(rules),
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Notification rules retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Notification rules retrieval failed: {str(e)}"
        )


@router.post(
    "/rules/evaluate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Rules evaluated successfully"},
        400: {"description": "Invalid context data"},
        500: {"description": "Rule evaluation error"}
    },
    summary="Evaluate notification rules",
    description="Evaluate notification rules against a context and send notifications"
)
@timed("notification_rules_evaluate")
async def evaluate_notification_rules(
    context: Dict[str, Any],
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Evaluate notification rules against context"""
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Evaluate rules
        await notification_service.evaluate_and_send_rules(context)
        
        logger.info(
            "Notification rules evaluated",
            context_keys=list(context.keys()),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Rules evaluated and notifications queued if conditions were met",
            "context": context,
            "request_id": request_id,
            "evaluated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Notification rules evaluation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Notification rules evaluation failed: {str(e)}"
        )


# System Integration Routes

@router.post(
    "/system/health-alert",
    responses={
        200: {"description": "Health alert sent successfully"},
        500: {"description": "Health alert error"}
    },
    summary="Send system health alert",
    description="Send a system health alert notification"
)
@timed("system_health_alert")
async def send_system_health_alert(
    status: str = Query(..., description="System status"),
    issue_description: str = Query(..., description="Issue description"),
    affected_components: str = Query(..., description="Affected components"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Send system health alert"""
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Send health alert using template
        notification_id = await notification_service.send_template_notification(
            template_id="system_health_warning",
            recipient="admin@example.com",
            variables={
                "status": status,
                "issue_description": issue_description,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "affected_components": affected_components
            }
        )
        
        logger.info(
            "System health alert sent",
            notification_id=notification_id,
            status=status,
            request_id=request_id
        )
        
        return {
            "success": True,
            "notification_id": notification_id,
            "message": "System health alert sent successfully",
            "metadata": {
                "status": status,
                "issue_description": issue_description,
                "affected_components": affected_components
            },
            "request_id": request_id,
            "sent_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "System health alert failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System health alert failed: {str(e)}"
        )


@router.post(
    "/engagement/alert",
    responses={
        200: {"description": "Engagement alert sent successfully"},
        500: {"description": "Engagement alert error"}
    },
    summary="Send engagement alert",
    description="Send an engagement alert notification"
)
@timed("engagement_alert")
async def send_engagement_alert(
    post_title: str = Query(..., description="Post title"),
    engagement_rate: float = Query(..., description="Engagement rate"),
    views: int = Query(..., description="Number of views"),
    likes: int = Query(..., description="Number of likes"),
    shares: int = Query(..., description="Number of shares"),
    comments: int = Query(..., description="Number of comments"),
    alert_type: str = Query(..., description="Alert type (high or low)"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Send engagement alert"""
    
    # Validate alert type
    if alert_type not in ["high", "low"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Alert type must be 'high' or 'low'"
        )
    
    try:
        # Get notification service
        notification_service = get_notification_service()
        
        # Determine template based on alert type
        template_id = f"post_engagement_{alert_type}"
        
        # Send engagement alert using template
        notification_id = await notification_service.send_template_notification(
            template_id=template_id,
            recipient="admin@example.com",
            variables={
                "post_title": post_title,
                "engagement_rate": engagement_rate,
                "views": views,
                "likes": likes,
                "shares": shares,
                "comments": comments
            }
        )
        
        logger.info(
            "Engagement alert sent",
            notification_id=notification_id,
            alert_type=alert_type,
            engagement_rate=engagement_rate,
            request_id=request_id
        )
        
        return {
            "success": True,
            "notification_id": notification_id,
            "message": f"Engagement {alert_type} alert sent successfully",
            "metadata": {
                "post_title": post_title,
                "engagement_rate": engagement_rate,
                "views": views,
                "likes": likes,
                "shares": shares,
                "comments": comments,
                "alert_type": alert_type
            },
            "request_id": request_id,
            "sent_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Engagement alert failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engagement alert failed: {str(e)}"
        )


# Export router
__all__ = ["router"]































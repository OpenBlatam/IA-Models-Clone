"""
Automation API routes for Facebook Posts API
Intelligent automation, scheduling, and business process orchestration
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..core.config import get_settings
from ..api.schemas import ErrorResponse
from ..api.dependencies import get_request_id
from ..services.automation_service import (
    get_automation_service, AutomationType, AutomationStatus, TriggerType,
    AutomationRule, AutomationExecution
)
from ..services.security_service import get_security_service
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/automation", tags=["Automation"])

# Security scheme
security = HTTPBearer()


# Automation Rule Management Routes

@router.get(
    "/rules",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Automation rules retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Automation rules retrieval error"}
    },
    summary="Get automation rules",
    description="Get all automation rules"
)
@timed("automation_rules_list")
async def get_automation_rules(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get automation rules"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Get rules
        rules = await automation_service.get_automation_rules()
        
        logger.info(
            "Automation rules retrieved",
            user_id=payload.get("user_id"),
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Automation rules retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Automation rules retrieval failed: {str(e)}"
        )


@router.post(
    "/rules",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Automation rule created successfully"},
        400: {"description": "Invalid rule data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Automation rule creation error"}
    },
    summary="Create automation rule",
    description="Create a new automation rule"
)
@timed("automation_rule_create")
async def create_automation_rule(
    rule_data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Create automation rule"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate required fields
        required_fields = ["id", "name", "description", "automation_type", "trigger_type", "action_config"]
        for field in required_fields:
            if field not in rule_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )
        
        # Validate automation type
        valid_automation_types = [at.value for at in AutomationType]
        if rule_data["automation_type"] not in valid_automation_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid automation type. Valid types: {valid_automation_types}"
            )
        
        # Validate trigger type
        valid_trigger_types = [tt.value for tt in TriggerType]
        if rule_data["trigger_type"] not in valid_trigger_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid trigger type. Valid types: {valid_trigger_types}"
            )
        
        # Create automation rule
        rule = AutomationRule(
            id=rule_data["id"],
            name=rule_data["name"],
            description=rule_data["description"],
            automation_type=AutomationType(rule_data["automation_type"]),
            trigger_type=TriggerType(rule_data["trigger_type"]),
            trigger_config=rule_data.get("trigger_config", {}),
            action_config=rule_data["action_config"],
            conditions=rule_data.get("conditions", []),
            enabled=rule_data.get("enabled", True),
            priority=rule_data.get("priority", 0),
            metadata=rule_data.get("metadata", {})
        )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Create rule
        success = await automation_service.create_automation_rule(rule)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create automation rule"
            )
        
        logger.info(
            "Automation rule created",
            rule_id=rule.id,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Automation rule created successfully",
            "rule_id": rule.id,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "automation_type": rule.automation_type.value,
                "trigger_type": rule.trigger_type.value,
                "enabled": rule.enabled,
                "priority": rule.priority
            },
            "request_id": request_id,
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Automation rule creation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Automation rule creation failed: {str(e)}"
        )


# Event Management Routes

@router.post(
    "/events/emit",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Event emitted successfully"},
        400: {"description": "Invalid event data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Event emission error"}
    },
    summary="Emit event",
    description="Emit an event to trigger automation rules"
)
@timed("automation_event_emit")
async def emit_event(
    event_type: str = Query(..., description="Event type"),
    event_data: Dict[str, Any] = Query(..., description="Event data"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Emit event"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate event data
        if not event_type or not event_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Event type and event data are required"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Emit event
        await automation_service.emit_event(event_type, event_data)
        
        logger.info(
            "Event emitted",
            event_type=event_type,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Event emitted successfully",
            "event_type": event_type,
            "event_data": event_data,
            "request_id": request_id,
            "emitted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Event emission failed",
            event_type=event_type,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Event emission failed: {str(e)}"
        )


# Execution Management Routes

@router.get(
    "/executions",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Automation executions retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Automation executions retrieval error"}
    },
    summary="Get automation executions",
    description="Get automation rule executions"
)
@timed("automation_executions_list")
async def get_automation_executions(
    rule_id: Optional[str] = Query(None, description="Filter by rule ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get automation executions"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Get executions
        executions = await automation_service.get_automation_executions(rule_id)
        
        logger.info(
            "Automation executions retrieved",
            user_id=payload.get("user_id"),
            rule_id=rule_id,
            executions_count=len(executions),
            request_id=request_id
        )
        
        return {
            "success": True,
            "executions": executions,
            "total_count": len(executions),
            "filter": {"rule_id": rule_id} if rule_id else None,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Automation executions retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Automation executions retrieval failed: {str(e)}"
        )


# Scheduled Tasks Management Routes

@router.get(
    "/scheduled-tasks",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Scheduled tasks retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Scheduled tasks retrieval error"}
    },
    summary="Get scheduled tasks",
    description="Get all scheduled automation tasks"
)
@timed("automation_scheduled_tasks")
async def get_scheduled_tasks(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get scheduled tasks"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Get scheduled tasks
        tasks = await automation_service.get_scheduled_tasks()
        
        logger.info(
            "Scheduled tasks retrieved",
            user_id=payload.get("user_id"),
            tasks_count=len(tasks),
            request_id=request_id
        )
        
        return {
            "success": True,
            "scheduled_tasks": tasks,
            "total_count": len(tasks),
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Scheduled tasks retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scheduled tasks retrieval failed: {str(e)}"
        )


# Predefined Automation Routes

@router.post(
    "/trigger/high-engagement",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "High engagement event triggered successfully"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Event trigger error"}
    },
    summary="Trigger high engagement event",
    description="Trigger high engagement automation event"
)
@timed("automation_trigger_high_engagement")
async def trigger_high_engagement_event(
    post_id: str = Query(..., description="Post ID"),
    engagement_rate: float = Query(..., description="Engagement rate"),
    user_email: str = Query(..., description="User email"),
    views: int = Query(0, description="Number of views"),
    likes: int = Query(0, description="Number of likes"),
    shares: int = Query(0, description="Number of shares"),
    comments: int = Query(0, description="Number of comments"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Trigger high engagement event"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not post_id or not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Post ID and user email are required"
            )
        
        if not (0 <= engagement_rate <= 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Engagement rate must be between 0 and 1"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Prepare event data
        event_data = {
            "post_id": post_id,
            "engagement_rate": engagement_rate,
            "user_email": user_email,
            "views": views,
            "likes": likes,
            "shares": shares,
            "comments": comments,
            "triggered_by": payload.get("user_id"),
            "triggered_at": datetime.now().isoformat()
        }
        
        # Emit event
        await automation_service.emit_event("post_engagement_high", event_data)
        
        logger.info(
            "High engagement event triggered",
            post_id=post_id,
            engagement_rate=engagement_rate,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "High engagement event triggered successfully",
            "event_type": "post_engagement_high",
            "event_data": event_data,
            "request_id": request_id,
            "triggered_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "High engagement event trigger failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"High engagement event trigger failed: {str(e)}"
        )


@router.post(
    "/trigger/low-engagement",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Low engagement event triggered successfully"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Event trigger error"}
    },
    summary="Trigger low engagement event",
    description="Trigger low engagement automation event"
)
@timed("automation_trigger_low_engagement")
async def trigger_low_engagement_event(
    post_id: str = Query(..., description="Post ID"),
    engagement_rate: float = Query(..., description="Engagement rate"),
    user_email: str = Query(..., description="User email"),
    views: int = Query(0, description="Number of views"),
    likes: int = Query(0, description="Number of likes"),
    shares: int = Query(0, description="Number of shares"),
    comments: int = Query(0, description="Number of comments"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Trigger low engagement event"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not post_id or not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Post ID and user email are required"
            )
        
        if not (0 <= engagement_rate <= 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Engagement rate must be between 0 and 1"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Prepare event data
        event_data = {
            "post_id": post_id,
            "engagement_rate": engagement_rate,
            "user_email": user_email,
            "views": views,
            "likes": likes,
            "shares": shares,
            "comments": comments,
            "triggered_by": payload.get("user_id"),
            "triggered_at": datetime.now().isoformat()
        }
        
        # Emit event
        await automation_service.emit_event("post_engagement_low", event_data)
        
        logger.info(
            "Low engagement event triggered",
            post_id=post_id,
            engagement_rate=engagement_rate,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Low engagement event triggered successfully",
            "event_type": "post_engagement_low",
            "event_data": event_data,
            "request_id": request_id,
            "triggered_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Low engagement event trigger failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Low engagement event trigger failed: {str(e)}"
        )


@router.post(
    "/trigger/post-published",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Post published event triggered successfully"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Event trigger error"}
    },
    summary="Trigger post published event",
    description="Trigger post published automation event"
)
@timed("automation_trigger_post_published")
async def trigger_post_published_event(
    post_id: str = Query(..., description="Post ID"),
    content: str = Query(..., description="Post content"),
    user_id: str = Query(..., description="User ID"),
    platform: str = Query("facebook", description="Platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Trigger post published event"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not all([post_id, content, user_id]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Post ID, content, and user ID are required"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Prepare event data
        event_data = {
            "post_id": post_id,
            "content": content,
            "user_id": user_id,
            "platform": platform,
            "published_at": datetime.now().isoformat(),
            "triggered_by": payload.get("user_id")
        }
        
        # Emit event
        await automation_service.emit_event("post_published", event_data)
        
        logger.info(
            "Post published event triggered",
            post_id=post_id,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Post published event triggered successfully",
            "event_type": "post_published",
            "event_data": event_data,
            "request_id": request_id,
            "triggered_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Post published event trigger failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Post published event trigger failed: {str(e)}"
        )


@router.post(
    "/trigger/user-registered",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "User registered event triggered successfully"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Event trigger error"}
    },
    summary="Trigger user registered event",
    description="Trigger user registered automation event"
)
@timed("automation_trigger_user_registered")
async def trigger_user_registered_event(
    user_id: str = Query(..., description="User ID"),
    user_email: str = Query(..., description="User email"),
    username: str = Query(..., description="Username"),
    registration_source: str = Query("api", description="Registration source"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Trigger user registered event"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not all([user_id, user_email, username]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID, email, and username are required"
            )
        
        # Get automation service
        automation_service = get_automation_service()
        
        # Prepare event data
        event_data = {
            "user_id": user_id,
            "user_email": user_email,
            "username": username,
            "registration_source": registration_source,
            "registered_at": datetime.now().isoformat(),
            "triggered_by": payload.get("user_id")
        }
        
        # Emit event
        await automation_service.emit_event("user_registered", event_data)
        
        logger.info(
            "User registered event triggered",
            user_id=user_id,
            user_email=user_email,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "User registered event triggered successfully",
            "event_type": "user_registered",
            "event_data": event_data,
            "request_id": request_id,
            "triggered_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "User registered event trigger failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User registered event trigger failed: {str(e)}"
        )


# Export router
__all__ = ["router"]






























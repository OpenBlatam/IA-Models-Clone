"""
Workflow API routes for Facebook Posts API
Automated workflows, content pipelines, and business process automation
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
from ..services.workflow_service import (
    get_workflow_service, WorkflowStatus, WorkflowTrigger, StepStatus,
    WorkflowStep, Workflow, WorkflowExecution
)
from ..services.security_service import get_security_service
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/workflows", tags=["Workflows"])

# Security scheme
security = HTTPBearer()


# Workflow Management Routes

@router.get(
    "/",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Workflows retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Workflow retrieval error"}
    },
    summary="Get available workflows",
    description="Get list of all available workflows"
)
@timed("workflow_list")
async def get_available_workflows(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get available workflows"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get workflow service
        workflow_service = get_workflow_service()
        
        # Get workflows
        workflows = await workflow_service.get_available_workflows()
        
        logger.info(
            "Workflows retrieved",
            user_id=payload.get("user_id"),
            workflows_count=len(workflows),
            request_id=request_id
        )
        
        return {
            "success": True,
            "workflows": workflows,
            "total_count": len(workflows),
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Workflow retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow retrieval failed: {str(e)}"
        )


@router.post(
    "/execute/{workflow_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Workflow execution started successfully"},
        400: {"description": "Invalid workflow or parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Workflow execution error"}
    },
    summary="Execute workflow",
    description="Execute a workflow with provided parameters"
)
@timed("workflow_execution")
async def execute_workflow(
    workflow_id: str = Path(..., description="Workflow ID to execute"),
    trigger_data: Dict[str, Any] = Query(..., description="Trigger data for workflow"),
    context: Optional[Dict[str, Any]] = Query(None, description="Additional context data"),
    executed_by: Optional[str] = Query(None, description="User executing the workflow"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Execute workflow"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get workflow service
        workflow_service = get_workflow_service()
        
        # Execute workflow
        execution_id = await workflow_service.execute_workflow(
            workflow_id=workflow_id,
            trigger_data=trigger_data,
            context=context,
            executed_by=executed_by or payload.get("user_id")
        )
        
        logger.info(
            "Workflow execution started",
            workflow_id=workflow_id,
            execution_id=execution_id,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Workflow execution started successfully",
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "running",
            "request_id": request_id,
            "started_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Workflow execution failed",
            workflow_id=workflow_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )


@router.get(
    "/status/{execution_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Workflow status retrieved successfully"},
        404: {"description": "Workflow execution not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Status retrieval error"}
    },
    summary="Get workflow execution status",
    description="Get the status of a workflow execution"
)
@timed("workflow_status")
async def get_workflow_status(
    execution_id: str = Path(..., description="Workflow execution ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get workflow execution status"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get workflow service
        workflow_service = get_workflow_service()
        
        # Get status
        status = await workflow_service.get_workflow_status(execution_id)
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow execution not found"
            )
        
        logger.info(
            "Workflow status retrieved",
            execution_id=execution_id,
            status=status["status"],
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "execution": status,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Workflow status retrieval failed",
            execution_id=execution_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow status retrieval failed: {str(e)}"
        )


@router.post(
    "/cancel/{execution_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Workflow cancelled successfully"},
        404: {"description": "Workflow execution not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Workflow cancellation error"}
    },
    summary="Cancel workflow execution",
    description="Cancel a running workflow execution"
)
@timed("workflow_cancellation")
async def cancel_workflow(
    execution_id: str = Path(..., description="Workflow execution ID to cancel"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Cancel workflow execution"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get workflow service
        workflow_service = get_workflow_service()
        
        # Cancel workflow
        success = await workflow_service.cancel_workflow(execution_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow execution not found"
            )
        
        logger.info(
            "Workflow execution cancelled",
            execution_id=execution_id,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Workflow execution cancelled successfully",
            "execution_id": execution_id,
            "request_id": request_id,
            "cancelled_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Workflow cancellation failed",
            execution_id=execution_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow cancellation failed: {str(e)}"
        )


# Predefined Workflow Execution Routes

@router.post(
    "/execute/content-creation",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content creation workflow started successfully"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Workflow execution error"}
    },
    summary="Execute content creation workflow",
    description="Execute the content creation and optimization pipeline"
)
@timed("content_creation_workflow")
async def execute_content_creation_workflow(
    topic: str = Query(..., description="Content topic"),
    audience_type: str = Query(..., description="Target audience type"),
    content_type: str = Query(..., description="Content type"),
    tone: str = Query("professional", description="Content tone"),
    optimization_strategy: str = Query("engagement", description="Optimization strategy"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Execute content creation workflow"""
    
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
        valid_audience_types = ["professionals", "general", "students"]
        valid_content_types = ["educational", "entertainment", "news", "promotional"]
        
        if audience_type not in valid_audience_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audience type. Valid types: {valid_audience_types}"
            )
        
        if content_type not in valid_content_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type. Valid types: {valid_content_types}"
            )
        
        # Get workflow service
        workflow_service = get_workflow_service()
        
        # Prepare trigger data
        trigger_data = {
            "topic": topic,
            "audience_type": audience_type,
            "content_type": content_type,
            "tone": tone,
            "optimization_strategy": optimization_strategy
        }
        
        # Execute workflow
        execution_id = await workflow_service.execute_workflow(
            workflow_id="content_creation_pipeline",
            trigger_data=trigger_data,
            executed_by=payload.get("user_id")
        )
        
        logger.info(
            "Content creation workflow started",
            execution_id=execution_id,
            topic=topic,
            audience_type=audience_type,
            content_type=content_type,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Content creation workflow started successfully",
            "execution_id": execution_id,
            "workflow_id": "content_creation_pipeline",
            "parameters": trigger_data,
            "request_id": request_id,
            "started_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Content creation workflow failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content creation workflow failed: {str(e)}"
        )


@router.post(
    "/execute/engagement-monitoring",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Engagement monitoring workflow started successfully"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Workflow execution error"}
    },
    summary="Execute engagement monitoring workflow",
    description="Execute the engagement monitoring and alerting workflow"
)
@timed("engagement_monitoring_workflow")
async def execute_engagement_monitoring_workflow(
    post_id: str = Query(..., description="Post ID to monitor"),
    user_email: str = Query(..., description="User email for notifications"),
    engagement_threshold: float = Query(0.8, description="Engagement threshold for alerts"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Execute engagement monitoring workflow"""
    
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
        
        if not (0 <= engagement_threshold <= 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Engagement threshold must be between 0 and 1"
            )
        
        # Get workflow service
        workflow_service = get_workflow_service()
        
        # Prepare trigger data
        trigger_data = {
            "post_id": post_id,
            "user_email": user_email,
            "engagement_threshold": engagement_threshold
        }
        
        # Execute workflow
        execution_id = await workflow_service.execute_workflow(
            workflow_id="engagement_monitoring",
            trigger_data=trigger_data,
            executed_by=payload.get("user_id")
        )
        
        logger.info(
            "Engagement monitoring workflow started",
            execution_id=execution_id,
            post_id=post_id,
            user_email=user_email,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Engagement monitoring workflow started successfully",
            "execution_id": execution_id,
            "workflow_id": "engagement_monitoring",
            "parameters": trigger_data,
            "request_id": request_id,
            "started_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Engagement monitoring workflow failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engagement monitoring workflow failed: {str(e)}"
        )


@router.post(
    "/execute/content-optimization",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content optimization workflow started successfully"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Workflow execution error"}
    },
    summary="Execute content optimization workflow",
    description="Execute the content optimization loop workflow"
)
@timed("content_optimization_workflow")
async def execute_content_optimization_workflow(
    user_id: str = Query(..., description="User ID for optimization"),
    optimization_strategy: str = Query("engagement", description="Optimization strategy"),
    max_posts: int = Query(10, description="Maximum number of posts to optimize"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Execute content optimization workflow"""
    
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
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        if max_posts <= 0 or max_posts > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Max posts must be between 1 and 100"
            )
        
        # Get workflow service
        workflow_service = get_workflow_service()
        
        # Prepare trigger data
        trigger_data = {
            "user_id": user_id,
            "optimization_strategy": optimization_strategy,
            "max_posts": max_posts
        }
        
        # Execute workflow
        execution_id = await workflow_service.execute_workflow(
            workflow_id="content_optimization_loop",
            trigger_data=trigger_data,
            executed_by=payload.get("user_id")
        )
        
        logger.info(
            "Content optimization workflow started",
            execution_id=execution_id,
            user_id=user_id,
            optimization_strategy=optimization_strategy,
            max_posts=max_posts,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Content optimization workflow started successfully",
            "execution_id": execution_id,
            "workflow_id": "content_optimization_loop",
            "parameters": trigger_data,
            "request_id": request_id,
            "started_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Content optimization workflow failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content optimization workflow failed: {str(e)}"
        )


# Workflow Template Routes

@router.get(
    "/templates",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Workflow templates retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Template retrieval error"}
    },
    summary="Get workflow templates",
    description="Get predefined workflow templates"
)
@timed("workflow_templates")
async def get_workflow_templates(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get workflow templates"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Define workflow templates
        templates = [
            {
                "id": "content_creation_pipeline",
                "name": "Content Creation Pipeline",
                "description": "Automated content creation and optimization pipeline",
                "parameters": [
                    {"name": "topic", "type": "string", "required": True, "description": "Content topic"},
                    {"name": "audience_type", "type": "string", "required": True, "description": "Target audience type"},
                    {"name": "content_type", "type": "string", "required": True, "description": "Content type"},
                    {"name": "tone", "type": "string", "required": False, "description": "Content tone", "default": "professional"},
                    {"name": "optimization_strategy", "type": "string", "required": False, "description": "Optimization strategy", "default": "engagement"}
                ],
                "steps": [
                    "Generate content using AI",
                    "Analyze generated content",
                    "Optimize content for better performance",
                    "Schedule optimized content for posting"
                ]
            },
            {
                "id": "engagement_monitoring",
                "name": "Engagement Monitoring",
                "description": "Monitor post engagement and send alerts",
                "parameters": [
                    {"name": "post_id", "type": "string", "required": True, "description": "Post ID to monitor"},
                    {"name": "user_email", "type": "string", "required": True, "description": "User email for notifications"},
                    {"name": "engagement_threshold", "type": "float", "required": False, "description": "Engagement threshold for alerts", "default": 0.8}
                ],
                "steps": [
                    "Track post analytics",
                    "Check if engagement is high",
                    "Send high engagement alert"
                ]
            },
            {
                "id": "content_optimization_loop",
                "name": "Content Optimization Loop",
                "description": "Continuously optimize content based on performance",
                "parameters": [
                    {"name": "user_id", "type": "string", "required": True, "description": "User ID for optimization"},
                    {"name": "optimization_strategy", "type": "string", "required": False, "description": "Optimization strategy", "default": "engagement"},
                    {"name": "max_posts", "type": "integer", "required": False, "description": "Maximum number of posts to optimize", "default": 10}
                ],
                "steps": [
                    "Get posts for optimization",
                    "Loop through posts and optimize",
                    "Update optimized posts"
                ]
            }
        ]
        
        logger.info(
            "Workflow templates retrieved",
            user_id=payload.get("user_id"),
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Workflow template retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow template retrieval failed: {str(e)}"
        )


# Export router
__all__ = ["router"]






























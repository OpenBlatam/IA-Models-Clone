"""
Workflow API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ....services.workflow_service import WorkflowService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_name: str = Field(..., description="Name of the workflow to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Workflow context data")


class WorkflowStepRequest(BaseModel):
    """Request model for workflow step definition."""
    name: str = Field(..., description="Step name")
    description: str = Field(..., description="Step description")
    function_name: str = Field(..., description="Function to execute")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Step dependencies")
    timeout: int = Field(default=300, ge=1, le=3600, description="Step timeout in seconds")


class CustomWorkflowRequest(BaseModel):
    """Request model for custom workflow creation."""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[WorkflowStepRequest] = Field(..., description="Workflow steps")


async def get_workflow_service(session: DatabaseSessionDep) -> WorkflowService:
    """Get workflow service instance."""
    return WorkflowService(session)


@router.post("/execute", response_model=Dict[str, Any])
async def execute_workflow(
    request: WorkflowExecutionRequest = Depends(),
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Execute a workflow with given context."""
    try:
        result = await workflow_service.execute_workflow(
            workflow_name=request.workflow_name,
            context=request.context,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Workflow '{request.workflow_name}' executed successfully"
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
            detail="Failed to execute workflow"
        )


@router.get("/available", response_model=Dict[str, Any])
async def get_available_workflows(
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Get available workflows."""
    try:
        workflows = workflow_service.get_available_workflows()
        
        return {
            "success": True,
            "data": workflows,
            "message": "Available workflows retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available workflows"
        )


@router.get("/executions", response_model=Dict[str, Any])
async def get_workflow_executions(
    workflow_name: Optional[str] = Query(None, description="Filter by workflow name"),
    status: Optional[str] = Query(None, description="Filter by execution status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of executions to return"),
    offset: int = Query(default=0, ge=0, description="Number of executions to skip"),
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Get workflow execution history."""
    try:
        executions = await workflow_service.get_workflow_executions(
            workflow_name=workflow_name,
            status=status,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "data": executions,
            "message": "Workflow executions retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow executions"
        )


@router.get("/executions/{execution_id}", response_model=Dict[str, Any])
async def get_workflow_execution_details(
    execution_id: int,
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Get detailed information about a specific workflow execution."""
    try:
        # Get execution details
        executions = await workflow_service.get_workflow_executions(
            limit=1,
            offset=0
        )
        
        # Find the specific execution
        execution = None
        for exec in executions["executions"]:
            if exec["id"] == execution_id:
                execution = exec
                break
        
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow execution not found"
            )
        
        return {
            "success": True,
            "data": execution,
            "message": "Workflow execution details retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow execution details"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_workflow_stats(
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Get workflow statistics."""
    try:
        stats = await workflow_service.get_workflow_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Workflow statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow statistics"
        )


@router.post("/content-publishing", response_model=Dict[str, Any])
async def execute_content_publishing_workflow(
    post_id: int = Query(..., description="Post ID to publish"),
    publish_at: Optional[str] = Query(None, description="Publish date (ISO format)"),
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Execute content publishing workflow for a specific post."""
    try:
        from datetime import datetime
        
        context = {
            "post_id": post_id,
            "publish_at": datetime.fromisoformat(publish_at) if publish_at else datetime.utcnow()
        }
        
        result = await workflow_service.execute_workflow(
            workflow_name="content_publishing",
            context=context,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content publishing workflow executed successfully"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
        )
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
            detail="Failed to execute content publishing workflow"
        )


@router.post("/content-moderation", response_model=Dict[str, Any])
async def execute_content_moderation_workflow(
    post_id: int = Query(..., description="Post ID to moderate"),
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Execute content moderation workflow for a specific post."""
    try:
        context = {
            "post_id": post_id
        }
        
        result = await workflow_service.execute_workflow(
            workflow_name="content_moderation",
            context=context,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content moderation workflow executed successfully"
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
            detail="Failed to execute content moderation workflow"
        )


@router.post("/user-onboarding", response_model=Dict[str, Any])
async def execute_user_onboarding_workflow(
    user_id: str = Query(..., description="User ID to onboard"),
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Execute user onboarding workflow for a specific user."""
    try:
        context = {
            "user_id": user_id
        }
        
        result = await workflow_service.execute_workflow(
            workflow_name="user_onboarding",
            context=context,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": "User onboarding workflow executed successfully"
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
            detail="Failed to execute user onboarding workflow"
        )


@router.get("/templates", response_model=Dict[str, Any])
async def get_workflow_templates():
    """Get workflow templates and examples."""
    templates = {
        "content_publishing": {
            "name": "Content Publishing",
            "description": "Automated workflow for publishing blog posts",
            "steps": [
                "validate_content",
                "generate_seo_metadata",
                "schedule_publication",
                "notify_subscribers"
            ],
            "use_cases": ["Blog post publishing", "Content approval", "SEO optimization"]
        },
        "content_moderation": {
            "name": "Content Moderation",
            "description": "Automated content moderation and quality assessment",
            "steps": [
                "spam_detection",
                "toxicity_analysis",
                "quality_assessment",
                "moderation_decision"
            ],
            "use_cases": ["Content review", "Spam detection", "Quality control"]
        },
        "user_onboarding": {
            "name": "User Onboarding",
            "description": "Automated user onboarding process",
            "steps": [
                "send_welcome_email",
                "create_user_profile",
                "recommend_content",
                "setup_notifications"
            ],
            "use_cases": ["New user registration", "User activation", "Profile setup"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "templates": templates,
            "total": len(templates)
        },
        "message": "Workflow templates retrieved successfully"
    }


@router.get("/status/{execution_id}", response_model=Dict[str, Any])
async def get_workflow_execution_status(
    execution_id: int,
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Get the current status of a workflow execution."""
    try:
        # Get execution details
        executions = await workflow_service.get_workflow_executions(
            limit=1,
            offset=0
        )
        
        # Find the specific execution
        execution = None
        for exec in executions["executions"]:
            if exec["id"] == execution_id:
                execution = exec
                break
        
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow execution not found"
            )
        
        status_info = {
            "execution_id": execution_id,
            "workflow_name": execution["workflow_name"],
            "status": execution["status"],
            "started_at": execution["started_at"],
            "completed_at": execution["completed_at"],
            "progress": "100%" if execution["status"] == "completed" else "50%" if execution["status"] == "running" else "0%"
        }
        
        return {
            "success": True,
            "data": status_info,
            "message": "Workflow execution status retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow execution status"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_workflow_performance_metrics(
    workflow_service: WorkflowService = Depends(get_workflow_service),
    current_user: CurrentUserDep = Depends()
):
    """Get workflow performance metrics."""
    try:
        # Get workflow stats
        stats = await workflow_service.get_workflow_stats()
        
        # Calculate performance metrics (mock implementation)
        performance_metrics = {
            "total_executions": stats["total_executions"],
            "success_rate": 0.85,  # Would be calculated from actual data
            "average_execution_time": 120.5,  # seconds
            "most_used_workflow": "content_publishing",
            "execution_trends": {
                "last_24h": 15,
                "last_7d": 85,
                "last_30d": 320
            },
            "error_rate": 0.05,
            "average_steps_per_execution": 3.2
        }
        
        return {
            "success": True,
            "data": performance_metrics,
            "message": "Workflow performance metrics retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow performance metrics"
        )


























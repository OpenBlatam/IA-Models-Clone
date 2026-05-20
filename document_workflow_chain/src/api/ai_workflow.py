"""
AI Workflow API - Advanced Implementation
========================================

Advanced AI workflow API with intelligent automation and AI-powered processing.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime

from ..services import ai_workflow_service, AIWorkflowType, AIWorkflowStatus

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class AIWorkflowCreateRequest(BaseModel):
    """AI workflow create request model"""
    name: str
    workflow_type: str
    input_data: Dict[str, Any]
    template_name: Optional[str] = None
    ai_provider: str = "openai"
    parameters: Optional[Dict[str, Any]] = None


class AIWorkflowResponse(BaseModel):
    """AI workflow response model"""
    workflow_id: str
    name: str
    type: str
    status: str
    created_at: str
    message: str


class AIWorkflowInfoResponse(BaseModel):
    """AI workflow info response model"""
    id: str
    name: str
    type: str
    status: str
    input_data: Dict[str, Any]
    steps: List[Dict[str, Any]]
    ai_provider: str
    parameters: Dict[str, Any]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    results: Dict[str, Any]
    errors: List[str]
    current_step: int
    progress: float


class AIWorkflowTemplateResponse(BaseModel):
    """AI workflow template response model"""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    ai_provider: str
    estimated_time: int


class AIWorkflowStatsResponse(BaseModel):
    """AI workflow statistics response model"""
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    ai_processing_time: float
    workflows_by_type: Dict[str, int]
    available_templates: int
    processing_queue_size: int


# AI Workflow creation endpoints
@router.post("/ai-workflows", response_model=AIWorkflowResponse)
async def create_ai_workflow(request: AIWorkflowCreateRequest):
    """Create AI-powered workflow"""
    try:
        # Validate workflow type
        try:
            workflow_type = AIWorkflowType(request.workflow_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid workflow type: {request.workflow_type}"
            )
        
        workflow_id = await ai_workflow_service.create_ai_workflow(
            name=request.name,
            workflow_type=workflow_type,
            input_data=request.input_data,
            template_name=request.template_name,
            ai_provider=request.ai_provider,
            parameters=request.parameters
        )
        
        return AIWorkflowResponse(
            workflow_id=workflow_id,
            name=request.name,
            type=request.workflow_type,
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            message="AI workflow created successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create AI workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create AI workflow: {str(e)}"
        )


@router.post("/ai-workflows/{workflow_id}/execute")
async def execute_ai_workflow(workflow_id: str):
    """Execute AI-powered workflow"""
    try:
        result = await ai_workflow_service.execute_ai_workflow(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "status": result["status"],
            "progress": result["progress"],
            "results": result["results"],
            "errors": result["errors"],
            "execution_time": result.get("execution_time", 0),
            "message": "AI workflow executed successfully" if result["status"] == "completed" else "AI workflow execution failed",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to execute AI workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute AI workflow: {str(e)}"
        )


# AI Workflow management endpoints
@router.get("/ai-workflows/{workflow_id}", response_model=AIWorkflowInfoResponse)
async def get_ai_workflow(workflow_id: str):
    """Get AI workflow information"""
    try:
        workflow = await ai_workflow_service.get_ai_workflow(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI workflow not found"
            )
        
        return AIWorkflowInfoResponse(**workflow)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AI workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI workflow: {str(e)}"
        )


@router.get("/ai-workflows", response_model=List[Dict[str, Any]])
async def list_ai_workflows(
    workflow_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """List AI workflows with filtering"""
    try:
        # Convert string parameters to enums
        type_enum = AIWorkflowType(workflow_type) if workflow_type else None
        status_enum = AIWorkflowStatus(status) if status else None
        
        workflows = await ai_workflow_service.list_ai_workflows(
            workflow_type=type_enum,
            status=status_enum,
            limit=limit
        )
        
        return workflows
    
    except Exception as e:
        logger.error(f"Failed to list AI workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list AI workflows: {str(e)}"
        )


@router.post("/ai-workflows/{workflow_id}/cancel")
async def cancel_ai_workflow(workflow_id: str):
    """Cancel AI workflow"""
    try:
        success = await ai_workflow_service.cancel_ai_workflow(workflow_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI workflow not found"
            )
        
        return {
            "workflow_id": workflow_id,
            "message": "AI workflow cancelled successfully",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel AI workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel AI workflow: {str(e)}"
        )


# AI Workflow templates endpoints
@router.get("/ai-workflows/templates", response_model=Dict[str, Any])
async def get_ai_workflow_templates():
    """Get available AI workflow templates"""
    try:
        templates = await ai_workflow_service.get_ai_workflow_templates()
        return templates
    
    except Exception as e:
        logger.error(f"Failed to get AI workflow templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI workflow templates: {str(e)}"
        )


@router.post("/ai-workflows/templates")
async def create_ai_workflow_template(
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
    ai_provider: str = "openai",
    estimated_time: int = 30
):
    """Create custom AI workflow template"""
    try:
        success = await ai_workflow_service.create_ai_workflow_template(
            name=name,
            description=description,
            steps=steps,
            ai_provider=ai_provider,
            estimated_time=estimated_time
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create AI workflow template"
            )
        
        return {
            "template_name": name,
            "message": "AI workflow template created successfully",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create AI workflow template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create AI workflow template: {str(e)}"
        )


# AI Workflow statistics endpoint
@router.get("/ai-workflows/stats", response_model=AIWorkflowStatsResponse)
async def get_ai_workflow_stats():
    """Get AI workflow service statistics"""
    try:
        stats = await ai_workflow_service.get_ai_workflow_stats()
        return AIWorkflowStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get AI workflow stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI workflow stats: {str(e)}"
        )


# Health check endpoint
@router.get("/ai-workflows/health")
async def ai_workflow_health():
    """AI workflow service health check"""
    try:
        stats = await ai_workflow_service.get_ai_workflow_stats()
        
        return {
            "service": "ai_workflow_service",
            "status": "healthy",
            "total_workflows": stats["total_workflows"],
            "processing_queue_size": stats["processing_queue_size"],
            "available_templates": stats["available_templates"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"AI workflow service health check failed: {e}")
        return {
            "service": "ai_workflow_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


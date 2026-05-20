"""
Workflows API Router
====================

API endpoints for workflow operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
import logging

from ..business_agents import BusinessArea
from ..workflow_engine import WorkflowStatus
from ..schemas.workflow_schemas import (
    WorkflowRequest, WorkflowResponse, WorkflowListResponse,
    WorkflowExecutionRequest, WorkflowExecutionResponse,
    WorkflowTemplateResponse, WorkflowExportResponse
)
from ..core.dependencies import get_workflow_service
from ..core.exceptions import convert_to_http_exception
from ..services.workflow_service import WorkflowService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Workflows"])

@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    status: Optional[WorkflowStatus] = Query(None, description="Filter by status"),
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """List all workflows with optional filters."""
    
    try:
        workflows_data = await workflow_service.list_workflows(
            business_area=business_area,
            created_by=created_by,
            status=status
        )
        
        return WorkflowListResponse(
            workflows=workflows_data,
            total=len(workflows_data),
            business_area=business_area.value if business_area else None,
            status=status.value if status else None
        )
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list workflows")

@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Get specific workflow details."""
    
    try:
        workflow_data = await workflow_service.get_workflow(workflow_id)
        return WorkflowResponse(**workflow_data)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get workflow")

@router.post("/", response_model=WorkflowResponse)
async def create_workflow(
    request: WorkflowRequest,
    created_by: str = Query(..., description="User creating the workflow"),
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Create a new workflow."""
    
    try:
        # Convert request to workflow service format
        steps = [
            {
                "name": step.name,
                "step_type": step.step_type.value,
                "description": step.description,
                "agent_type": step.agent_type,
                "parameters": step.parameters,
                "conditions": step.conditions,
                "next_steps": step.next_steps,
                "parallel_steps": step.parallel_steps,
                "max_retries": step.max_retries,
                "timeout": step.timeout
            }
            for step in request.steps
        ]
        
        workflow_data = await workflow_service.create_workflow(
            name=request.name,
            description=request.description,
            business_area=request.business_area,
            steps=steps,
            created_by=created_by,
            variables=request.variables
        )
        
        return WorkflowResponse(**workflow_data)
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Execute a workflow."""
    
    try:
        result = await workflow_service.execute_workflow(workflow_id)
        return WorkflowExecutionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute workflow")

@router.post("/{workflow_id}/pause", response_model=Dict[str, Any])
async def pause_workflow(
    workflow_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Pause a running workflow."""
    
    try:
        result = await workflow_service.pause_workflow(workflow_id)
        return result
        
    except Exception as e:
        logger.error(f"Failed to pause workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to pause workflow")

@router.post("/{workflow_id}/resume", response_model=Dict[str, Any])
async def resume_workflow(
    workflow_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Resume a paused workflow."""
    
    try:
        result = await workflow_service.resume_workflow(workflow_id)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to resume workflow")

@router.delete("/{workflow_id}", response_model=Dict[str, Any])
async def delete_workflow(
    workflow_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Delete a workflow."""
    
    try:
        result = await workflow_service.delete_workflow(workflow_id)
        return result
        
    except Exception as e:
        logger.error(f"Failed to delete workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete workflow")

@router.get("/{workflow_id}/export", response_model=WorkflowExportResponse)
async def export_workflow(
    workflow_id: str,
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Export workflow as JSON."""
    
    try:
        export_data = await workflow_service.export_workflow(workflow_id)
        return WorkflowExportResponse(
            workflow_data=export_data,
            exported_at=export_data.get("exported_at"),
            version="1.0"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to export workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export workflow")

@router.post("/import", response_model=WorkflowResponse)
async def import_workflow(
    workflow_data: Dict[str, Any],
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Import workflow from JSON."""
    
    try:
        result = await workflow_service.import_workflow(workflow_data)
        return WorkflowResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to import workflow: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to import workflow")

@router.get("/templates", response_model=Dict[str, List[WorkflowTemplateResponse]])
async def get_workflow_templates(
    workflow_service: WorkflowService = Depends(get_workflow_service)
):
    """Get predefined workflow templates for each business area."""
    
    try:
        templates_data = await workflow_service.get_workflow_templates()
        
        # Convert to response format
        result = {}
        for area, templates in templates_data.items():
            result[area] = [
                WorkflowTemplateResponse(**template) for template in templates
            ]
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get workflow templates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get workflow templates")

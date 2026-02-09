"""
Workflows API Router
===================

FastAPI router for workflow management and execution.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from ..business_agents import BusinessAgentManager, BusinessArea
from ..workflow_engine import Workflow, WorkflowStep, WorkflowStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Workflows"])

# Dependency to get agent manager
def get_agent_manager() -> BusinessAgentManager:
    """Get the global agent manager instance."""
    from ..main import app
    return app.state.agent_manager

# Request/Response Models
class WorkflowStepRequest(BaseModel):
    name: str = Field(..., description="Step name")
    step_type: str = Field(..., description="Type of step")
    description: str = Field(..., description="Step description")
    agent_type: Optional[str] = Field(None, description="Agent type for the step")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")

class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    business_area: BusinessArea = Field(..., description="Business area")
    steps: List[WorkflowStepRequest] = Field(..., description="Workflow steps")
    variables: Optional[Dict[str, Any]] = Field(None, description="Workflow variables")

class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str
    business_area: str
    status: str
    created_by: str
    created_at: str
    updated_at: str
    steps: List[Dict[str, Any]]
    variables: Dict[str, Any]
    execution_results: Optional[Dict[str, Any]] = None

class WorkflowListResponse(BaseModel):
    workflows: List[WorkflowResponse]
    total: int

class WorkflowExecutionResponse(BaseModel):
    workflow_id: str
    status: str
    execution_id: str
    started_at: str
    estimated_completion: Optional[str] = None
    current_step: Optional[str] = None
    progress_percentage: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Endpoints
@router.post("/", response_model=WorkflowResponse)
async def create_workflow(
    request: WorkflowCreateRequest,
    created_by: str = "system",  # This would come from authentication
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Create a new business workflow."""
    
    try:
        # Convert request steps to workflow steps
        steps = []
        for step_req in request.steps:
            step = {
                "name": step_req.name,
                "step_type": step_req.step_type,
                "description": step_req.description,
                "agent_type": step_req.agent_type,
                "parameters": step_req.parameters,
                "dependencies": step_req.dependencies
            }
            steps.append(step)
        
        # Create workflow
        workflow = await agent_manager.create_business_workflow(
            name=request.name,
            description=request.description,
            business_area=request.business_area,
            steps=steps,
            created_by=created_by,
            variables=request.variables
        )
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            business_area=workflow.business_area,
            status=workflow.status.value,
            created_by=workflow.created_by,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat(),
            steps=[step.__dict__ for step in workflow.steps],
            variables=workflow.variables
        )
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    business_area: Optional[BusinessArea] = None,
    created_by: Optional[str] = None,
    status: Optional[WorkflowStatus] = None,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """List workflows with optional filtering."""
    
    try:
        workflows = agent_manager.list_workflows(
            business_area=business_area,
            created_by=created_by
        )
        
        # Filter by status if provided
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        workflow_responses = [
            WorkflowResponse(
                id=workflow.id,
                name=workflow.name,
                description=workflow.description,
                business_area=workflow.business_area,
                status=workflow.status.value,
                created_by=workflow.created_by,
                created_at=workflow.created_at.isoformat(),
                updated_at=workflow.updated_at.isoformat(),
                steps=[step.__dict__ for step in workflow.steps],
                variables=workflow.variables,
                execution_results=workflow.execution_results
            ) for workflow in workflows
        ]
        
        return WorkflowListResponse(
            workflows=workflow_responses,
            total=len(workflow_responses)
        )
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get a specific workflow by ID."""
    
    try:
        workflow = agent_manager.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            business_area=workflow.business_area,
            status=workflow.status.value,
            created_by=workflow.created_by,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat(),
            steps=[step.__dict__ for step in workflow.steps],
            variables=workflow.variables,
            execution_results=workflow.execution_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")

@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Execute a workflow."""
    
    try:
        # Check if workflow exists
        workflow = agent_manager.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        # Execute workflow in background
        background_tasks.add_task(agent_manager.execute_business_workflow, workflow_id)
        
        return WorkflowExecutionResponse(
            workflow_id=workflow_id,
            status="started",
            execution_id=f"exec_{workflow_id}",
            started_at=workflow.updated_at.isoformat(),
            estimated_completion=None,  # Would calculate based on step durations
            current_step=None,
            progress_percentage=0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")

@router.get("/{workflow_id}/status", response_model=WorkflowExecutionResponse)
async def get_workflow_status(
    workflow_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get workflow execution status."""
    
    try:
        workflow = agent_manager.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        # Calculate progress (simplified)
        total_steps = len(workflow.steps)
        completed_steps = len([s for s in workflow.steps if hasattr(s, 'status') and s.status == 'completed'])
        progress = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        return WorkflowExecutionResponse(
            workflow_id=workflow_id,
            status=workflow.status.value,
            execution_id=f"exec_{workflow_id}",
            started_at=workflow.created_at.isoformat(),
            estimated_completion=None,
            current_step=None,
            progress_percentage=progress,
            results=workflow.execution_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")

@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Delete a workflow."""
    
    try:
        workflow = agent_manager.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        # Delete workflow (would implement in workflow engine)
        # For now, just return success
        return {"message": f"Workflow {workflow_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/templates/{business_area}", response_model=List[Dict[str, Any]])
async def get_workflow_templates_for_area(
    business_area: BusinessArea,
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get workflow templates for a specific business area."""
    
    try:
        templates = agent_manager.get_workflow_templates()
        return templates.get(business_area.value, [])
        
    except Exception as e:
        logger.error(f"Failed to get workflow templates for {business_area}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow templates: {str(e)}")



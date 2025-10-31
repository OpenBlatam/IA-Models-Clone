"""
Workflow Schemas
================

Pydantic models for workflow operations.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..business_agents import BusinessArea
from ..workflow_engine import StepType, WorkflowStatus

class WorkflowStepRequest(BaseModel):
    """Request schema for workflow step creation."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Step name")
    step_type: StepType = Field(..., description="Step type")
    description: str = Field(..., min_length=1, max_length=500, description="Step description")
    agent_type: str = Field(..., description="Agent type for this step")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Step conditions")
    next_steps: List[str] = Field(default_factory=list, description="Next step IDs")
    parallel_steps: List[str] = Field(default_factory=list, description="Parallel step IDs")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    timeout: int = Field(300, ge=1, le=3600, description="Step timeout in seconds")

class WorkflowRequest(BaseModel):
    """Request schema for workflow creation."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Workflow name")
    description: str = Field(..., min_length=1, max_length=1000, description="Workflow description")
    business_area: BusinessArea = Field(..., description="Business area")
    steps: List[WorkflowStepRequest] = Field(..., description="Workflow steps")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class WorkflowStepResponse(BaseModel):
    """Response schema for workflow step."""
    
    id: str = Field(..., description="Step ID")
    name: str = Field(..., description="Step name")
    step_type: str = Field(..., description="Step type")
    description: str = Field(..., description="Step description")
    agent_type: str = Field(..., description="Agent type")
    parameters: Dict[str, Any] = Field(..., description="Step parameters")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Step conditions")
    next_steps: List[str] = Field(..., description="Next step IDs")
    parallel_steps: List[str] = Field(..., description="Parallel step IDs")
    max_retries: int = Field(..., description="Maximum retry attempts")
    timeout: int = Field(..., description="Step timeout in seconds")
    status: str = Field(..., description="Step status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class WorkflowResponse(BaseModel):
    """Response schema for workflow."""
    
    id: str = Field(..., description="Workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    business_area: str = Field(..., description="Business area")
    status: str = Field(..., description="Workflow status")
    created_by: str = Field(..., description="Creator user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    variables: Dict[str, Any] = Field(..., description="Workflow variables")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    steps: List[WorkflowStepResponse] = Field(..., description="Workflow steps")

class WorkflowListResponse(BaseModel):
    """Response schema for workflow list."""
    
    workflows: List[WorkflowResponse] = Field(..., description="List of workflows")
    total: int = Field(..., description="Total number of workflows")
    business_area: Optional[str] = Field(None, description="Filtered business area")
    status: Optional[str] = Field(None, description="Filtered status")

class WorkflowExecutionRequest(BaseModel):
    """Request schema for workflow execution."""
    
    workflow_id: str = Field(..., description="Workflow ID to execute")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data for execution")
    variables: Optional[Dict[str, Any]] = Field(None, description="Execution variables")

class WorkflowExecutionResponse(BaseModel):
    """Response schema for workflow execution."""
    
    workflow_id: str = Field(..., description="Workflow ID")
    status: str = Field(..., description="Execution status")
    execution_results: Optional[Dict[str, Any]] = Field(None, description="Execution results")
    error: Optional[str] = Field(None, description="Error message if failed")
    executed_at: datetime = Field(..., description="Execution timestamp")
    duration: Optional[float] = Field(None, description="Execution duration in seconds")

class WorkflowTemplateResponse(BaseModel):
    """Response schema for workflow template."""
    
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    business_area: str = Field(..., description="Business area")
    steps: List[Dict[str, Any]] = Field(..., description="Template steps")
    category: Optional[str] = Field(None, description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    is_public: bool = Field(..., description="Whether template is public")
    usage_count: int = Field(..., description="Usage count")
    rating: float = Field(..., description="Template rating")

class WorkflowExportResponse(BaseModel):
    """Response schema for workflow export."""
    
    workflow_data: Dict[str, Any] = Field(..., description="Exported workflow data")
    exported_at: datetime = Field(default_factory=datetime.now, description="Export timestamp")
    version: str = Field("1.0", description="Export format version")
